"""H100 jobs: prompt embeddings, dense C1 references, and reduced-budget conditions scored in-job (protocol sections 6-8).

A reference job loads (or, the first time, generates and saves) its seed's starting noise, runs dense C1 with the
registered sampler, decodes, and keeps the lossless frames (PNG) and the registered MP4. A condition job runs one
condition on the same saved noise and embeddings, scores masked, outside-ROI and full-frame LPIPS and the descriptive
pixel checks against the reference's lossless frames under the reference's measurement masks, and writes the MP4.

``WanBackend`` is the real model. ``DryRunBackend`` runs the same condition code with tiny random models at a small
latent size, to exercise it, and hands on stand-in full-size frames: a stock clip for each reference, and for each
condition that clip with a seeded disturbance whose size inside and outside the ROI is set per condition. The dry
run's numbers say nothing about Wan; they only exercise the code that will produce them.
"""

import time

import numpy as np
import torch

from density import noise as noise_module
from density import prompts as prompt_set
from density.conditions import lookup, run_condition, sha256
from density.masks.measurement import measurement_masks
from density.masks.storage import load_masks
from density.metrics import fidelity
from density.run import plan
from density.run.manifest import Job
from density.run.store import Run, file_sha256, load_frames, save_frames

FULL_LATENT_SHAPE = (1, 16, 21, 60, 104)
BUDGET_TOLERANCE = 0.005  # section 11: the actual token-step fraction within 0.005 of the registered target


def registered_fraction(condition: str, preset: str | None) -> float:
    """The registered token-step fraction: 1 for C1, 20/40 for C2, and the preset's at the full 32,760 tokens."""
    from density.hsa import PRESETS, preset_iterations

    if condition == "C1":
        return 1.0
    if condition == "C2":
        return 0.5
    p = PRESETS[preset]
    rows = preset_iterations(p)
    return (p.high_tokens * int(rows[0].sum()) + p.low_tokens * int(rows[1].sum())) / (p.num_tokens * rows.shape[1])


def check_budget(job, fraction: float) -> dict:
    target = registered_fraction(job.condition, job.preset)
    if abs(fraction - target) > BUDGET_TOLERANCE:
        raise RuntimeError(f"{job.id}: token-step fraction {fraction:.6f} is not within {BUDGET_TOLERANCE} of the registered {target:.6f}")
    return {"actual": fraction, "registered": target, "tolerance": BUDGET_TOLERANCE}


def to_uint8(frames: np.ndarray) -> np.ndarray:
    """WanPipeline's [0, 1] frames to uint8, exactly as the references, the canary and the smoke test quantize them."""
    return (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)


class WanBackend:
    """Wan2.1-T2V-1.3B from the verified weights: the bf16 Transformer, the fp32 VAE, and UMT5 for embeddings."""

    dry_run = False
    latent_shape = FULL_LATENT_SHAPE
    weights_info = None

    def __init__(self, device: str, weights_dir):
        self.device, self.weights = device, weights_dir
        self._transformer = self._vae = self._text = None

    def verify(self, components):
        from density import wan_weights

        verified = wan_weights.verify(self.weights, components)
        self.weights_info = {"repo": verified["repo"], "revision": verified["revision"], "files_verified": sorted(verified["files"])}
        return verified

    def embed(self, texts: dict[str, str]) -> dict[str, torch.Tensor]:
        """UMT5 in bf16 (registered), loaded once for a session's embedding jobs and released before generation."""
        from transformers import AutoTokenizer, UMT5EncoderModel

        from density import text_encoding

        if self._text is None:
            self.verify(("text_encoder", "tokenizer"))
            tokenizer = AutoTokenizer.from_pretrained(self.weights / "tokenizer")
            encoder = UMT5EncoderModel.from_pretrained(self.weights / "text_encoder", dtype=text_encoding.T5_DTYPE, device_map=self.device).eval()
            self._text = (tokenizer, encoder)
        return text_encoding.encode(texts, *self._text, self.device)

    def release_text_encoder(self) -> None:
        if self._text is not None:
            self._text = None
            import gc

            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

    def models(self):
        self.release_text_encoder()
        if self._transformer is None:
            from diffusers import AutoencoderKLWan, WanTransformer3DModel

            self.verify(("transformer", "vae"))
            self._transformer = WanTransformer3DModel.from_pretrained(self.weights / "transformer", torch_dtype=torch.bfloat16).to(self.device).eval()
            self._vae = AutoencoderKLWan.from_pretrained(self.weights / "vae", torch_dtype=torch.float32).to(self.device).eval()
        return self._transformer, self._vae

    def decode(self, latents: torch.Tensor) -> np.ndarray:
        from diffusers.video_processor import VideoProcessor

        from density.wan import decode_latents

        _, vae = self.models()
        video = decode_latents(vae, latents.to(self.device))
        return to_uint8(VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="np")[0])

    def coverage(self, m: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(m.astype(np.float32)).flatten()

    def frames(self, job: Job, decoded: np.ndarray, reference: np.ndarray | None, roi: np.ndarray | None, latents_sha256: str) -> np.ndarray:
        return decoded


DRY_RUN_CLIPS = {"face": "face", "manipulation": "hands", "body": "body", "control": "landscape"}
# Seeded disturbance per condition: (sigma inside the ROI, sigma outside it), in uint8 levels, before smoothing.
DRY_RUN_DISTURBANCE = {
    ("C2", None): (5.5, 4.0), ("C3", None): (4.0, 2.5), ("C5", None): (3.7, 2.7), ("C6", None): (3.6, 2.75), ("A1", None): (3.6, 2.55),
    ("C4", 1.0): (3.8, 2.52), ("C4", 3.0): (3.4, 2.55), ("C4", 10.0): (3.2, 2.9),
    ("C7", None): (4.6, 2.4),  # the exploratory inverse (density.exploratory): a stand-in, worse inside the region
}


class DryRunBackend:
    """Tiny random models at a small latent size, with stand-in full-size frames (see the module docstring)."""

    dry_run = True
    weights_info = {"repo": "dry run: tiny random models", "revision": None}

    def __init__(self, device: str = "cpu", noise_prompts: tuple[int, ...] = ()):
        from density.checks import euler

        self.device = device
        self.latent_shape = euler.TINY_LATENT_SHAPE
        self.noise_prompts = set(noise_prompts)  # prompts whose reference is the tiny model's own output, so their masks fail
        self._transformer = self._vae = None
        self._clips: dict[str, np.ndarray] = {}

    def verify(self, components):
        return None

    def embed(self, texts):
        from density import text_encoding
        from density.checks import smoke

        tokenizer, encoder = smoke.tiny_text_encoder(self.device)
        return text_encoding.encode(texts, tokenizer, encoder, self.device)

    def models(self):
        if self._transformer is None:
            from diffusers import AutoencoderKLWan

            from density.checks import euler
            from density.checks import hsa as hsa_checks

            self._transformer = euler.tiny_transformer(torch.bfloat16, self.device, hsa_checks.TINY_CONFIG)
            torch.manual_seed(euler.SEED)
            self._vae = AutoencoderKLWan(base_dim=8, z_dim=16, num_res_blocks=1).to(self.device).eval()
        return self._transformer, self._vae

    def decode(self, latents):
        from diffusers.video_processor import VideoProcessor

        from density.wan import decode_latents

        _, vae = self.models()
        video = decode_latents(vae, latents.to(self.device))
        return to_uint8(VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="np")[0])

    def coverage(self, m: np.ndarray) -> torch.Tensor:
        """Full-size coverage (21, 30, 52) reduced to the tiny grid by max-pooling, so the dry run ranks real ROIs."""
        _, _, f, h, w = self.latent_shape
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        return torch.nn.functional.adaptive_max_pool3d(t, (f, h // 2, w // 2))[0, 0].flatten()

    def clip(self, stratum: str) -> np.ndarray:
        from density.masks.clips import load_frames as load_clip

        name = DRY_RUN_CLIPS[stratum]
        if name not in self._clips:
            self._clips[name] = load_clip(name)
        return self._clips[name]

    def frames(self, job: Job, decoded: np.ndarray, reference: np.ndarray | None, roi: np.ndarray | None, latents_sha256: str) -> np.ndarray:
        """Stand-ins: the reference with a disturbance drawn from the run's own final latents, so identical generations give
        identical frames (an empty-mask C4 equals C3, as section 11 requires); C3's sizes apply wherever the ROI is empty."""
        if job.condition == "C1":
            if job.prompt_id in self.noise_prompts:  # the tiny model's own output, enlarged to full size: no face, no hands
                up = decoded[np.arange(81) * len(decoded) // 81]
                return np.ascontiguousarray(np.repeat(np.repeat(up, 4, axis=1), 7, axis=2)[:, :480, :832])
            return self.clip(job.stratum).copy()
        inside = roi if roi is not None else np.zeros(reference.shape[:3], bool)
        key = (job.condition, job.lam if job.condition == "C4" else None) if inside.any() else ("C3", None)
        sigma_in, sigma_out = DRY_RUN_DISTURBANCE[key]
        if job.preset in ("HSA-75B",):
            sigma_in, sigma_out = 0.6 * sigma_in, 0.6 * sigma_out
        scale = np.random.default_rng([job.prompt_id, job.seed]).lognormal(0.0, 0.25)  # shared by a prompt-seed's conditions
        rng = np.random.default_rng(int(latents_sha256[:16], 16))
        sigma = np.where(inside[..., None], sigma_in, sigma_out) * scale
        noise = rng.standard_normal(reference.shape) * sigma
        return np.clip(reference.astype(np.float64) + noise, 0, 255).round().astype(np.uint8)


# Inputs


def embedding_path(run: Run, prompt_id: int):
    return run.path("inputs", "embeddings", f"{prompt_id:03d}.safetensors")


def noise_path(run: Run, prompt_id: int, seed: int):
    return run.path("inputs", "noise", f"{prompt_id:03d}-s{seed}.safetensors")


def run_embed(run: Run, job: Job, backend, rows: dict[int, dict]) -> dict:
    from safetensors.torch import save_file

    from density import text_encoding
    from density.run.store import write_bytes_atomic

    text = rows[job.prompt_id]["text"]
    texts = {"prompt": text, "negative": text_encoding.NEGATIVE_PROMPT}
    start = time.perf_counter()
    embeds = backend.embed(texts)
    path = embedding_path(run, job.prompt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file({k: v.contiguous() for k, v in embeds.items()}, str(tmp))
    tmp.replace(path)
    return {"text": text, "file": str(path.relative_to(run.folder)), "file_sha256": file_sha256(path),
            "sha256": {k: sha256(v) for k, v in embeds.items()}, "shape": list(embeds["prompt"].shape), "seconds": time.perf_counter() - start}


def load_embeddings(run: Run, job: Job, device: str):
    from safetensors.torch import load_file

    record = run.record(f"embed-{job.prompt_id:03d}")
    embeds = load_file(str(run.path(record["file"])))
    for k in ("prompt", "negative"):
        if sha256(embeds[k]) != record["sha256"][k]:
            raise RuntimeError(f"embed-{job.prompt_id:03d}: the saved {k} embedding does not match its record")
    return embeds["prompt"].to(device), embeds["negative"].to(device), record["sha256"]


def load_or_create_noise(run: Run, job: Job, backend) -> tuple[torch.Tensor, dict]:
    """The seed's starting noise (registered, density.noise): generated once on CPU and saved by the first reference job
    of the prompt and seed, then always loaded and checked by the hash that job recorded. Both pilot settings share it."""
    path = noise_path(run, job.prompt_id, job.seed)
    for other in run.jobs():
        if other.kind == "reference" and other.prompt_id == job.prompt_id and other.seed == job.seed and (rec := run.record(other)):
            return noise_module.load(path, rec["noise"]["sha256"]), rec["noise"]
    if job.kind != "reference":
        raise RuntimeError(f"{job.id}: no reference of prompt {job.prompt_id} seed {job.seed} has a record, so there is no registered noise")
    if job.seed not in prompt_set.seeds(job.prompt_id):
        raise ValueError(f"{job.id}: seed {job.seed} is not one of the prompt's planned seeds {prompt_set.seeds(job.prompt_id)}")
    if path.exists():  # saved by an attempt that failed before its record: the same generation, kept rather than redone
        from safetensors.torch import load_file

        latents = load_file(str(path))[noise_module.KEY]
        digest = sha256(latents)
        origin = "an earlier attempt of this job"
    else:
        latents = noise_module.generate(job.seed, backend.latent_shape)
        digest = noise_module.save(latents, path)
        origin = "this job"
    if tuple(latents.shape) != tuple(backend.latent_shape):
        raise RuntimeError(f"{path}: noise shape {tuple(latents.shape)}, this run needs {backend.latent_shape}")
    import platform

    return latents, {"file": str(path.relative_to(run.folder)), "sha256": digest, "seed": job.seed, "shape": list(latents.shape),
                     "generated_by": origin, "generated_on_cpu": platform.machine()}


def reference_inputs(run: Run, job: Job) -> tuple[np.ndarray, dict, np.ndarray]:
    """The reference's lossless frames (checked), its record, and its ROI."""
    ref_id = f"ref-{job.prompt_id:03d}-s{job.seed}-{job.setting}"
    record = run.record(ref_id)
    frames = load_frames(run.job_dir(ref_id) / record["frames"]["folder"], record["frames"])
    roi = load_masks(run.mask_dir(job.group))["ROI"]
    return frames, record, roi


def coverage_inputs(run: Run, job: Job, backend) -> dict[str, torch.Tensor]:
    """The token coverage the condition's rule reads, from the reference's mask job."""
    from density.conditions import HANDS_CONTACT, ROI

    condition = lookup(job.condition)
    if condition.mask is None:
        return {}
    stored = np.load(run.mask_dir(job.group) / "coverage.npz")
    key = {ROI: "ROI", HANDS_CONTACT: "A1"}[condition.mask]
    if key not in stored:
        raise RuntimeError(f"{job.id}: the mask job stored no {key} coverage")
    return {condition.mask: backend.coverage(stored[key])}


def generate(job: Job, backend, inputs, coverage) -> dict:
    """Run the job's condition and decode it, with CUDA-event timing on CUDA."""
    from density.checks.smoke import Timer, memory, reset_peak

    latents, prompt_embeds, negative_embeds = inputs
    setting = plan.SETTINGS[job.setting]
    transformer, _ = backend.models()
    device = backend.device
    reset_peak(device)
    start = time.perf_counter()
    with Timer(device) as denoise:
        result = run_condition(
            job.condition, transformer, prompt_id=job.prompt_id, latents=latents.to(device), seed=job.seed,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, guidance_scale=setting["guidance"], shift=setting["shift"],
            preset=job.preset or "HSA-50", lam=job.lam, coverage=coverage,
        )
    peak_denoise = memory(device)
    active = None
    if result.hsa is not None:  # section 8: every iteration, layer and branch ran the schedule's count, with the same IDs in both branches
        from density.checks.smoke import active_counts_ok

        active = active_counts_ok(result)
        result.hsa.caches.clear()
        result.hsa = None
        if not (active["counts_match"] and active["same_ids_both_branches"]):
            raise RuntimeError(f"{job.id}: active-token counts or IDs differ from the schedule: {({k: v for k, v in active.items() if k != 'active_counts'})}")
    with Timer(device) as decode_timer:
        decoded = backend.decode(result.latents)
    return {"decoded": decoded, "result": result, "active": active,
            "timing": {"denoising_seconds": denoise.seconds, "decode_seconds": decode_timer.seconds, "end_to_end_seconds": time.perf_counter() - start},
            "memory": {"denoising": peak_denoise, "end": memory(device)}, "latents_sha256": sha256(result.latents)}


def run_reference(run: Run, job: Job, backend, rows) -> dict:
    from density import video

    latents, noise_record = load_or_create_noise(run, job, backend)
    prompt_embeds, negative_embeds, embed_hashes = load_embeddings(run, job, backend.device)
    out = generate(job, backend, (latents, prompt_embeds, negative_embeds), {})
    frames = backend.frames(job, out["decoded"], None, None, out["latents_sha256"])
    folder = run.job_dir(job)
    saved = save_frames(frames, folder / "frames")
    mp4 = video.encode(frames, folder / "video.mp4")
    allocation = out["result"].record.summary()
    return {"noise": noise_record, "embeddings_sha256": embed_hashes, "setting": plan.SETTINGS[job.setting], "frames": saved,
            "video": {**mp4, "path": "video.mp4"}, "latents_sha256": out["latents_sha256"], "allocation": allocation,
            "budget": check_budget(job, allocation["token_step_fraction"]), "weights": backend.weights_info,
            "timing": out["timing"], "memory": out["memory"]}


def run_condition_job(run: Run, job: Job, backend, rows, lpips_model) -> dict:
    """One reduced-budget condition: generate, score in-job against the reference's lossless frames, then keep the
    lossless copy (FFV1, checked bit for bit) for the slice's quality and detector scores, and the playback MP4."""
    from density import video
    from density.run.store import write_json_atomic

    reference, ref_record, roi = reference_inputs(run, job)
    latents, noise_record = load_or_create_noise(run, job, backend)
    prompt_embeds, negative_embeds, embed_hashes = load_embeddings(run, job, backend.device)
    if embed_hashes != ref_record["embeddings_sha256"] or noise_record["sha256"] != ref_record["noise"]["sha256"]:
        raise RuntimeError(f"{job.id}: its inputs differ from its reference's")
    coverage = coverage_inputs(run, job, backend)
    out = generate(job, backend, (latents, prompt_embeds, negative_embeds), coverage)
    frames = backend.frames(job, out["decoded"], reference, roi, out["latents_sha256"])
    if frames.shape != reference.shape or frames.dtype != np.uint8:
        raise RuntimeError(f"{job.id}: frames {frames.shape} {frames.dtype} do not match the reference {reference.shape}")
    measurement = measurement_masks(roi)
    regions = None
    if job.stratum == "manipulation":  # A1's hands and contact, and the held object, scored separately (section 3)
        stored = load_masks(run.mask_dir(job.group))
        regions = {"HC": stored["H"] | stored["C"], "O": stored["O"]}
    start = time.perf_counter()
    scores = fidelity.fidelity(lpips_model, frames, reference, measurement, regions)
    scoring_seconds = time.perf_counter() - start
    fidelity.check_required(scores, [n for n in fidelity.required(job.stratum) if n in scores])
    folder = run.job_dir(job)
    lossless = video.encode_lossless(frames, folder / "frames.mkv")
    mp4 = video.encode(frames, folder / "video.mp4")
    allocation = out["result"].record.summary()
    budget = check_budget(job, allocation["token_step_fraction"])
    write_json_atomic(folder / "allocation.json", allocation)
    record = {"noise": noise_record, "embeddings_sha256": embed_hashes, "setting": plan.SETTINGS[job.setting],
              "reference": {"job": ref_record["job"]["id"], "uint8_sha256": ref_record["frames"]["uint8_sha256"]},
              "masks_sha256": {"masks.json": file_sha256(run.mask_dir(job.group) / "masks.json"), "coverage.npz": file_sha256(run.mask_dir(job.group) / "coverage.npz")},
              "fidelity": scores, "fidelity_seconds": scoring_seconds, "lossless": lossless, "video": {**mp4, "path": "video.mp4"},
              "latents_sha256": out["latents_sha256"], "token_step_fraction": allocation["token_step_fraction"], "budget": budget,
              "weights": backend.weights_info, "active_counts": out["active"],
              "allocation_file": "allocation.json", "allocation_diagnostics": allocation["diagnostics"],
              "group_tokens": allocation.get("group_tokens"), "timing": out["timing"], "memory": out["memory"]}
    return record
