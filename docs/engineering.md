# Density Phase 0: engineering notes

The project's working notebook: setup, every milestone, the GPU pods and the pilot's procedure. The overview is the
[README](../README.md). The code in this repository is written with Claude, Anthropic's AI assistant, under my direction.

Start here on the MacBook. The first task is to check the allocation rules with small, fixed test inputs. The HSA author's reply is not needed for these checks.

## Folder guide

| Folder | What goes here |
| --- | --- |
| `docs/` | A copy of the revised protocol and research notes. |
| `configs/` | Saved test settings, such as grid size, schedules, and fixed random seeds. |
| `src/` | The program that calculates scores, selects token groups, and maps masks. |
| `tests/` | Checks with known expected answers. |
| `data/sample_clips/` | Stock test clips for the mask check, with their conformed frames and masks (not versioned; see `configs/test_clips.json`). |
| `data/test_masks/` | Small region masks used to check the mapping. |
| `results/checks/` | Pass/fail reports, the mask audit CSV, and mask previews. |
| `scripts/` | One-time weight conversions to data-only files; `scripts/pod/` runs the smoke test and the experiment sessions on RunPod. |
| `third_party/` | Vendored code with its licenses and recorded patches: the Hands23 detector, and VBench's five quality dimensions with CLIP, DINO and MUSIQ. |
| `results/runs/` | Pilot and main runs pulled from the volume (not versioned). |
| `scratch/dryrun/` | The Mac's dry runs (not versioned); a real run refuses to read them. |

## Milestone 1: Euler sampler (passed)

The custom FlowMatch Euler loop in `src/density/sampler.py` matches diffusers'
`FlowMatchEulerDiscreteScheduler` bitwise at shifts 8 and 5 with 40 and 20 steps,
on CPU and MPS. The comparison is driven by a synthetic velocity field at the full
Wan latent shape and by a tiny random Wan Transformer (fp32 and bf16) running
through the stock `WanPipeline`. `src/density/checks/euler.py` describes each check.

Notes for the registration:

- diffusers applies the shift twice to the schedule's endpoint, so the last
  nonzero sigma is 0.0602 at shift 8 and 0.0244 at shift 5. The report records
  every sigma.
- Build the scheduler as `FlowMatchEulerDiscreteScheduler(shift=...)`. The Wan
  checkpoint ships a UniPC config with `flow_shift: 3.0`; `from_config` on it
  silently gives shift 1.
- The stock bf16 `WanPipeline` does guidance in bf16 and keeps latents in bf16
  between steps. The custom loop keeps both in fp32, as the protocol requires.
- Generate every condition through this loop, including the C1 and C2 references.
  On the tiny test model, the stock bf16 pipeline drifts from the fp32 path by about
  as much as a wrong schedule does, so mixing the two would bias every fidelity score.

### Real-model check on the Mac (passed)

`src/density/checks/wan_real.py` runs the real Wan2.1-T2V-1.3B (revision
`0fad780a`, bf16 Transformer, fp32 VAE) on MPS at 416 x 240 with 17 frames, using
placeholder text embeddings, shift 8, guidance 6, and 40 steps.

- Outputs are finite, and a repeat run matches bitwise.
- The stock pipeline, given fp32 velocities, matches the custom loop bitwise.
  `decode_latents` matches the pipeline's own decode bitwise.
- The decoded clip is coherent video, not noise.
- The stock bf16 pipeline differs from the custom loop by 33.9 dB decoded PSNR,
  below the protocol's 45 dB repeat-run threshold. Mixing the two code paths
  would fail the protocol's own correctness test.
- One 40-step run takes about 104 s on an M4 Pro.

MPS repeatability does not carry over to CUDA; the GPU smoke test re-measures it.

## Milestone 2: HSA mechanics (passed)

`src/density/hsa.py` implements HSA's compute mechanism on the validated Euler loop.
The group label of each token is an input; the need score and the C3-C6 ranking
rules are the next milestone.

- **Schedule.** Protocol equation 13 with half-open windows. At N = 32,760 the
  registered presets give low-group evaluation counts 16, 25, 19, 19 and
  token-step fractions 0.496007, 0.748752, 0.5, 0.5.
- **Active-token forward.** Only active tokens pass through the blocks. It reuses
  diffusers' patch embedding, condition embedder, RoPE (gathered by token ID),
  blocks and output head; only the self-attention processor is swapped.
- **KV caches.** One cache of K (after q/k norm and RoPE) and V per layer and
  guidance branch. Active tokens refresh their rows; active queries attend to
  every row. Reading a row before it is written raises an error.
- **Cached Euler update.** Each token keeps its last guided velocity in fp32;
  every token advances by dt times its fresh or cached velocity.

`src/density/checks/hsa.py` checks T0-T7 against a masked-dense oracle built on
diffusers' own forward pass. Self-attention is the only operation that mixes
video tokens; the report cites the source lines, and T0 confirms it empirically.

- Tiny random Transformer at the real attention geometry: every test bitwise in
  fp32 and bf16 on CPU and MPS, and every negative control fails at the step it
  should.
- Real Transformer on MPS: T1 matches the dense loop and the committed R1 frame
  hashes, and the final acceptance gate (T3, 3 label seeds) is bitwise.

The tiny fixture changed during this milestone. The first fixture (12-dim heads,
8 text tokens) failed T3 once by one bf16 rounding step, because the attention
kernel's accumulation order depends on the number of queries at that head size.
The fixture now matches the real model's 128-dim heads and 512 text tokens. The
report keeps the head-dim-12 results and the kernel evidence as a diagnostic.

On CUDA, attention and matmul kernels may not be row-invariant at full
resolution, so the GPU smoke test must re-measure fast-path-versus-oracle
equality before any tolerance is set.

## Milestone 3: allocation rules (passed)

The allocation side of protocol v2, on top of the HSA mechanics. Masks are synthetic
here; the detector and SAM 2 pipeline is Milestone 4.

- `src/density/allocation.py`: the need score (equation 3) from the fp32 guided
  velocities of the K leading iterations in which every token runs, read from the
  schedule (K = 5, 10, 9, 12), the C3-C6 rankings with their registered tie rules,
  and group assignment. Labels are assigned once, after iteration K - 1, then frozen.
- `src/density/token_map.py`: pixel masks to fractional token coverage
  (equations 10-12).
- `src/density/diagnostics.py`: precision, recall, normalized recall, lift, A_ROI,
  S_ROI and the no-op test (equations 17-20), with N/A for an empty ROI.
- `src/density/conditions.py`: one declarative table for C1-C6 and A1, and
  `run_condition`, which returns the output and an allocation record (g, m, w,
  scores, labels, active iterations, diagnostics, input hashes).

`src/density/checks/allocation.py` checks T8-T18 at full size with synthetic
inputs, on the tiny Transformer (CPU and MPS, fp32 and bf16), and on the real
Transformer. It also runs the sphere demo, which is informational only.

T12 as first written ("masked tokens can only rise in rank relative to C3") is
false when coverage is fractional: a more-covered masked token can overtake a
less-covered one. The report gives a counterexample. It judges the invariants
that hold for any coverage, and judges the literal statement for uniform
coverage, where it holds.

## Milestone 4: mask pipeline (checks passed; contact sheets await sign-off)

`src/density/masks/` builds the region masks of protocol section 4 and the
measurement masks of section 7 for one 81-frame, 832 x 480 reference clip.
Every registered value and rule is in `src/density/masks/registered.py`.

- **Regions.** F from SCRFD faces, H from WiLoR hands (with handedness), O and
  the contact state from Hands23 on keyframes 0, 8, ..., 80, and B from Mask
  R-CNN R50-FPN-3x people. Each detector's boxes prompt SAM 2.1 Hiera-small,
  which propagates them through the clip, forward and backward.
- **SAM 2.1** runs in fp32 without autocast, the same way on the Mac and the
  pod. A pre-hook casts SAM's bfloat16 memory to fp32 at the memory-attention
  input, and sam-2 is built without its CUDA extension.
- **Association (registered 2026-10-06).** Detections are matched to SAM tracks
  by `tracking.Association`:
  - A track is re-prompted when its IoU with a fresh detection falls below 0.5,
    but only by a detection that overlaps it (IoU > 0) and has the same hand
    side. A lost track (empty mask) is the one exception: it can be
    re-prompted without overlap.
  - A detection at IoU >= 0.5 belongs to that track whatever its side label,
    so no track starts on top of another. Side disagreements are logged.
  - SAM's masks before a track's first detection are kept, flagged in the
    metadata, and hatched on the contact sheets.
  - After tracking, a hand track is kept only if a Hands23 hand box (either
    side) overlaps it at IoU >= 0.5 on at least one keyframe. Unconfirmed
    tracks are dropped from H, never merged, and outlined in grey on the sheets.
- **Contact.** Contact intervals need `object_contact` at adjacent keyframes
  with the same hand and object tracks (sustained contact: anything shorter
  than 0.5 s cannot validate). They are trimmed, never extrapolated, and C
  is defined on validated frames only. Every dilation uses one Euclidean disk.
- **Per reference.** Each reference computes only the regions its stratum
  uses: F for faces; H, O and C for manipulation; B for full body; nothing
  for controls. It gets:
  - the ROI for its stratum and the A1 mask;
  - token coverage from `token_map`;
  - measurement masks, and eligibility with ordered replacement;
  - RLE storage and metadata;
  - a contact sheet and a row in the audit CSV (`pipeline.py`).
- **Weights.** Every weight file loads as data only. The Mask R-CNN and WiLoR
  checkpoints, which need a full unpickle, were converted once by
  `scripts/convert_mask_rcnn.py` and `scripts/convert_wilor_detector.py`. The
  check re-verifies the WiLoR conversion bitwise.
- **Licensing.** Ultralytics (AGPL-3.0) is imported only in `masks/wilor.py`.
  Hands23 is vendored in `third_party/` with its LICENSE and the recorded patch
  `third_party/patches/hands23_detector.diff`.

Weights go under `weights/masks/` (the list, sources, licenses and hashes are in
`src/density/masks/weights.py`); nothing downloads at run time. Copy every file
in that list to the pod, including the WiLoR conversion-check reference under
`weights/masks/wilor/conversion_check/`, which the check verifies. The four stock
test clips are registered with their sources, licenses and segments in
`configs/test_clips.json`, and live under `data/sample_clips/<id>/source.webm`
(not versioned).

```bash
uv run python -m density.checks.masks
```

It conforms the clips, runs the logic tests and every clip on CPU, and writes:
- `results/checks/mask_pipeline.md`, `.json` and `.sha256`;
- the audit CSV `results/checks/mask_pipeline_audit.csv`. Reviewers accept or reject
  each reference by the audit criteria in the protocol's section 4: a reject names
  one of three kinds of mask invalidity and gives a written reason. Sign-offs carry
  over to reruns while a sheet's hash is unchanged;
- contact sheets under `results/checks/mask_pipeline/`, and masks under
  `data/sample_clips/<id>/masks/` (neither is versioned).

`--logic-only` runs only the model-free tests.

## Milestone 5: metrics, prompt set, runner and the frozen analysis (built; dry-run on the Mac)

Everything the pilot and the main run need, following protocol sections 6, 7, 9, 10 and 12. Nothing has run on a pod
yet. The check is `python -m density.checks.metrics all` (about 2.7 hours on the Mac, most of it the quality models on
CPU); it writes `results/checks/metrics.md` and `.json`.

- **Prompt set** (`src/density/prompts.py`, `src/density/prompt_draft.py`, `configs/prompt_sources.json`,
  `configs/prompts.csv`). Claude drafts every row by the procedure Chase registered on 2026-10-07 (seed 20261009, before
  any draw), and Chase reviews every row. His eligibility readings are registered with it (`READINGS`): for
  manipulation, one person's hands in sustained contact with an object the source names, and nothing else in the
  close-up.
  - `python -m density.prompt_draft order STRATUM` prints a stratum's shuffled candidates for judging. The judgments go
    into the unversioned `data/prompt_draft/judgments.csv`, which holds the source texts.
  - `python -m density.prompt_draft assemble` (`--partial` while a stratum's candidates run out) writes
    `configs/prompts.csv` and the draw log `configs/prompt_draw_log.csv`, which records every candidate up to the last
    one kept, with its reason and no source text. It also writes the review file `data/prompt_draft/review.csv` (source
    text, eligibility reason and draft per row, holdout first; not versioned).
  - Chase marks each row `approve`, `edit` (with `chase_text`) or `reject` (with `chase_note`). The next `assemble`
    applies his edits, refills a rejected slot with the next eligible candidate and keeps the rejection, and refuses to
    overwrite a row he reviewed.
  - Manipulation's first pass gave too few candidates, so a second pass follows it (registered before its draw):
    the 300 Kinetics-400 classes VBench's human_action prompts do not use, as "A person is {class}", ranked 501-800
    with seed [20261009, 4]. `python -m density.prompt_draft derive-kinetics` writes that list from the CVDF's
    `val.csv` (CC BY 4.0) after checking its hash and its labels against torchvision's copy.
  - IDs are fixed by block (face holdout 1-6, manipulation holdout 7-12, face primary 13-48, manipulation primary
    49-84, body 85-100, controls 101-116, then the reserves from 117), integers 1-999 because equation 14's seeds are
    1000 + ID and 2000 + ID.
  - `python -m density.prompts validate` checks the whole file and `--holdout` the 12 holdout rows, which is all the
    pilot's init checks. The `subject`, `action` and `framing` columns let the validator check that each prompt names
    all three.
  - Only the rewritten prompts, with a citation, may ever be published (T2V-CompBench has no license).
- **Metrics** (`src/density/video.py`, `src/density/metrics/`; registered values in `metrics/registered.py`).
  - Every score is measured before lossy compression (registered by Chase, 2026-10-07). References keep their PNG
    frames. Each condition keeps an FFV1 copy (`frames.mkv`: Matroska, bgra with alpha 255, channels reordered in
    numpy), which must decode bit for bit before its job completes.
  - One fixed MP4 per video, the playback copy: libx264 CRF 10, yuv420p, 16 fps, one thread, cpu-independent, with
    BT.709 limited-range conversion in numpy. Each encode records the decoded MP4's PSNR against its lossless frames.
    Decoded frames, not files, are compared by hash; the encode canary (`configs/encode_canary.json`) checks them at
    the start of every H100 session.
  - LPIPS v0.1 (AlexNet), built data-only and checked equal to lpips' own loading: masked and outside-ROI LPIPS
    average the spatial map (equations 15-16); full-frame LPIPS is the scalar per frame, averaged (registered by Chase,
    2026-10-07). Masked PSNR and SSIM and crop LPIPS are descriptive. All run inside each H100 job, against the
    reference's lossless frames.
  - The five VBench dimensions, from VBench v0.1.5's own code vendored in `third_party/` (diffs in
    `third_party/patches/`), on the 0-100 scale: x100 for four dimensions, MUSIQ's per-video value as is. A NaN stops
    the job. Face identity (ArcFace), hand detection and box overlap are descriptive. These run on the slice, from the
    lossless frames (each FFV1 copy's hash checked first), with a prompt-seed's reference and conditions scored on one
    host. The MP4s are scored too, as a descriptive check of the encoding's effect; `analysis.collect` blocks the
    decision on any quality score not taken from lossless frames.
  - Weights: `src/density/metrics/weights.py` lists every file with its source, size, SHA-256 and license; all load as
    data only (CLIP and AMT-S were converted once by `scripts/convert_metric_weights.py`). AMT-S and MUSIQ are
    non-commercial: research use only.
- **Runner** (`src/density/run/`). A run folder on the volume holds an append-only manifest, one record per job
  (written last and atomically, so a stopped session resumes), per-job logs, the masks and sheets, the scores, the
  audit file and the freeze. Jobs enter the manifest in the protocol's order: embeddings, references and masks;
  replacement references after the review; conditions and scores only at the freeze, which hashes every reference,
  mask and audit decision, and every condition job re-checks those hashes. A session first passes the gates (strict
  determinism; the canary and the encode canary on the H100, the mask canary on the slice; the metric canary on both),
  checks the volume with du before each batch (the 112 GB stop), and pins each prompt's GPU model and image. Condition
  jobs assert section 8's active-token counts and section 11's budget; score jobs assert equal counts across
  conditions and, on controls, that C4 reproduces C3 exactly. `python -m density.run cost RUN` prints the cost table.
- **Metric canary** (`src/density/metric_canary.py`, registered by Chase on 2026-10-07): before any job, each session
  scores fixed inputs in a fresh process and must reproduce the registration bit for bit.
  - On the H100: the face clip's first 17 frames, MP4 against lossless, by LPIPS, PSNR, SSIM and crop LPIPS, plus that
    clip's FFV1 round trip.
  - On the slice: the quality dimensions of the lossless frames and of the MP4, and the three detectors.
  - A machine's first pilot session records the registration from two identical fresh processes. Pull it with
    `sync.sh registrations-pull`, copy it into `configs/metric_canary.json`, commit, and push the code before the
    next session.
- **Pilot rules** (`src/density/pilot.py`): the setting by detection coverage, lambda, the no-op stop, the C2 check
  (rounding tolerance only), pass rates with reserve sizing at the estimate (Chase sets the sizes after the pilot), and
  guardrail reachability. Failed holdout prompts are excluded and reported, never replaced.
- **Frozen analysis** (`src/density/analysis.py`): section 9 exactly, frozen by code hash in
  `configs/analysis_freeze.json`; it refuses to run if the code or numpy/scipy changed. Synthetic validations are in
  `src/density/checks/analysis_validation.py`.
- **Visual check** (`src/density/visual_check.py`): 30 C3/C4 pairs chosen at the freeze; blinded items, random sides
  and order, the key in a separate folder; `unblind` lists where the metric and the eye disagree.
- **Registration** (`src/density/registration.py`): every registered value with its file and the commit that last
  changed it, plus the pilot's slots.
- **Dry runs** (`src/density/checks/dryrun.py`) write only under `scratch/dryrun/`, which a real run refuses to read.
  Tiny random models stand in for Wan and stock clips for its frames, so their numbers describe nothing about Wan.

The pilot on the pods, step by step (each pod created only with Chase's approval):

0. Chase has approved the 12 holdout rows in `data/prompt_draft/review.csv`, and `assemble` lists none in
   `holdout_not_yet_approved`.
1. Push the code, weights, clips and prompt sources (`scripts/pod/sync.sh code|mask-weights|metric-weights|clips|prompt-sources`;
   `prompt-sources` goes with every code push, since validating `prompts.csv` reads them), and record the pod with
   `sync.sh pod-env UTC GPU IMAGE`. After a change to `uv.lock`, run `uv sync --locked` on the pod (with env.sh sourced).
2. On an H100: `sync.sh run-cmd init /workspace/runs/pilot --kind pilot` (it validates the holdout rows only), then
   `sync.sh run-exp h100 pilot`. This runs the canary, then the metric canary (which records on the first session), then
   the embeddings and the 24 references.
   - If the encode canary stops this first session, re-register it on the pod with `run-cmd encode-canary --record`
     (decision 7). Pull it with `sync.sh registrations-pull`, commit it with the metric canary's registration, push the
     code, and start the session again.
3. On the slice: `sync.sh run-exp slice pilot` (the mask canary, the metric canary, then the 24 mask jobs), then
   `run-cmd choose-setting` and `run-cmd audit /workspace/runs/pilot --setting S`; pull with `sync.sh run-pull pilot`
   and `sync.sh registrations-pull`, and commit `configs/metric_canary.json` before the next session.
4. Chase reviews the sheets in `results/runs/pilot/audit/audit.csv`; push it back with `sync.sh run-push-audit pilot`.
5. `run-cmd freeze`, then `run-exp h100 pilot` (conditions, LPIPS in-job) and `run-exp slice pilot` (quality and
   detector scores).
6. Pull, then on the Mac `python -m density.run pilot-rules results/runs/pilot` and
   `python -m density.registration build --out DIR --pilot results/runs/pilot`.

## GPU pod on RunPod

The RunPod MCP server runs only on the Mac (user scope in Claude Code). It manages
infrastructure only; work on a pod goes over SSH from the Mac.

### Standing rules

- **The volume:** the `density` network volume is never deleted, resized or modified.
  Claude Code's user settings deny the RunPod MCP tools that delete or update network
  volumes, and ask before any pod or cluster is created.
- **Creating pods:** ask Chase before creating any pod. At most one pod runs at a time,
  always on Secure Cloud and always with the density volume attached.
- **Deleting pods:** delete the pod, rather than stopping it, as soon as a session's jobs
  finish and the results are synced to the Mac. Deletion needs no confirmation.
- **Credentials:** never install the MCP on a pod, and never put RunPod or GitHub
  credentials there.

### Pod configuration (every session)

| Setting | Value |
| --- | --- |
| Cloud and GPU | Secure Cloud, 1 GPU (see "GPUs" below) |
| Data center | US-NE-1, the volume's data center |
| Storage | the `density` network volume, ID `n881ioz25j`, 125 GB (resized by Chase from 75 GB to 100 GB on 2026-10-06 and to 125 GB on 2026-10-07), Standard tier, created 2026-10-06, mounted at `/workspace` |
| Template | Runpod Pytorch 2.8.0 (`runpod-torch-v280`): `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, CUDA 12.8.1, host CUDA 12.8 to 13.4. This is the newest CUDA among RunPod's non-cluster PyTorch templates; the tag doesn't say "devel", so check `nvcc --version` on the first pod |
| Container disk | 30 GB |
| Ports | `22/tcp` only (no Jupyter) |
| SSH | `startSsh`, using the account's registered key, which is the Mac's `~/.ssh/id_ed25519.pub` |

### GPUs (US-NE-1)

- **Video generation** (the smoke test, the pilot and the main run): the H100 SXM 80 GB
  (`NVIDIA H100 80GB HBM3`), $3.49/hr on 2026-10-06. It was chosen on cost per
  reference, using the benchmark's relative times (RTX 4090 1.0, A100 0.6, H100 0.3).
- **Setup without GPU measurements** (weight downloads, the environment, the CUDA 13
  toolkit, the Detectron2 build): the cheapest pod in US-NE-1, the 24 GB MIG slice of an
  RTX PRO 6000 Blackwell (`NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb`),
  $0.59/hr on 2026-10-06. Delete it when setup is done, then create the H100 pod.
- **Masks:** the pipeline runs on that same 24 GB slice, never the H100 (registered as
  `MASK_GPU`), and so does the mask determinism check. SCRFD stays on the onnxruntime CPU
  provider.

### CUDA

- **torch:** torch 2.14.1 exactly as `uv.lock` pins it. Its Linux wheel is the CUDA 13.0
  build, with the CUDA 13.0.3 runtime libraries and cuDNN for CUDA 13. No CUDA 12.8 build
  of torch 2.14.1 exists.
- **Host driver:** pods require host CUDA 13.0 or newer (`gpu.minCudaVersion` "13.0").
- **Toolkit:** NVIDIA's CUDA 13.0 toolkit is installed once under `/workspace`. The
  template's own CUDA 12.8.1 toolkit is not used for our builds.
- **Detectron2** is built against that toolkit (`CUDA_HOME` set to it) with
  `TORCH_CUDA_ARCH_LIST="9.0;12.0"`, which covers the H100 (9.0) and the Blackwell slice
  (12.0).
- **sam-2** stays built without its CUDA extension (`SAM2_BUILD_CUDA=0`, registered).

### Session rules (every pod)

- **Caches on the volume:** the virtual environment, the uv cache, uv's Python installs and
  the Hugging Face cache live on `/workspace`:
  `UV_PROJECT_ENVIRONMENT=/workspace/venv`, `UV_CACHE_DIR=/workspace/uv-cache`,
  `UV_PYTHON_INSTALL_DIR=/workspace/python`, `HF_HOME=/workspace/hf`.
- **Code:** sync it from the Mac with rsync, including `.git`, into
  `/workspace/Density_Project` (excluding `.venv/`). There are no GitHub credentials on the
  pod.
- **Metadata:** every job's metadata records the GPU model and driver version
  (`density.checks.euler.gpu_info`, included in each report's `environment` block).
- **Long jobs and storage:** run long jobs in tmux. Anything outside `/workspace` is lost
  with the pod.
- **Results:** sync them back to the Mac with rsync, then delete the pod.

These rules override any later prompt that differs: sync by rsync only, and delete pods
rather than stopping them.

### Smoke test (protocol sections 8, 11, 12 and 13)

The smoke test measures what only CUDA can: exactness, determinism, memory, timing, and the
real mask cost at full resolution. It produces no experiment data. Its two prompts
(`configs/smoke_prompts.json`, seeds 9001 and 9002) are excluded from every prompt pool.

Each step is a subcommand of `python -m density.checks.smoke` that writes
`results/checks/smoke_test/<step>.json`. Steps are skipped if they already ran at the same
code SHA from the same inputs. The starting noise and the text embeddings are written once
and never recomputed. `report` runs on the Mac and writes `results/checks/smoke_test.md`,
`.json` and `.sha256`. The step folder itself isn't versioned.

| Session | Pod | Steps |
| --- | --- | --- |
| A | 24 GB slice | `bootstrap.sh`, then `weights`, `noise`, `encode`, `env` |
| B | H100 SXM | `env`, `weights`, `pytest`, `milestones`, `generate`, `sdpa`, `hsa_dense`, `t3`, `perf`, `vae_precision`, `sheet` |
| C | 24 GB slice | `env`, `weights`, `masks`, `mask_check`, `mask_compare` |
| D | a fresh H100 SXM | `env`, `weights`, `milestones`, `generate`, `sdpa`, `hsa_dense`, `t3`, `perf`, `sheet`, `crossmachine`, with flash attention registered, into `smoke_test/flash/` |
| E | 24 GB slice | `env`, `weights`, then `scripts/pod/diagnostics/mask_threads.py all`: the mask-speed diagnosis |
| F | 24 GB slice | `env`, `weights`, `mask_check`, `mask_compare` with the registered configuration, into `smoke_test/rerun/`; then `scripts/pod/rerun_gate.py` and the mask canary, recorded and checked |
| Mac | | `noise_mac` (the cross-device noise hashes), `report` |

- **Settings:** C1 at 832 x 480, 81 frames, shift 8, guidance 6, 40 steps. C4 runs at
  lambda 3 with a centered disk of radius 120 px on every frame.
- **Starting noise** (registered, `density/noise.py`):
  - Generated once on CPU with a seeded generator, saved, and identified by its SHA-256.
  - Always loaded from the saved file and never regenerated, because the same seed gives
    different noise on x86 and on ARM.
- **Text encoding:** T5 runs in bf16 with 512 tokens, using Wan2.1's own negative prompt.
  ftfy is absent, and that absence is registered (`density/text_encoding.py`).
- **Deterministic settings** (`density/determinism.py`):
  - cuDNN benchmarking is off.
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set by `scripts/pod/env.sh`.
  - Deterministic algorithms run in warn-only mode, and every warning is recorded.
  - The SDPA backend actually used is recorded from `torch._fused_sdp_choice` and checked
    against the profiler's kernel names.
- **Failures** stop the session: errors, out-of-memory errors and hash mismatches.
  - Cross-device hash comparisons are reported, not judged.
  - A same-device result that isn't bitwise is a measurement. It's reported relative to the
    repeat-run noise on that device.
  - A protocol threshold missed on the same device (repeat-run decoded PSNR below 45 dB) is
    flagged as blocking the pilot.
  - When the fast path and the oracle differ, `t3` reports the deviation as decoded PSNR.
    Dense repeats are bitwise on CUDA, so repeat-run noise is no yardstick there. `t3`
    also saves the fast-path and oracle frames losslessly, for the LPIPS check against the
    effect size in Milestone 5.
  - The bug test is structural. `t3` finds the first module that differs and checks that
    its inputs, the fast path's gathered RoPE rows and its skipped tokens' cache rows all
    equal the oracle's. Only then may kernel row-count dependence explain the difference,
    and otherwise it's a bug to investigate. C3 and C4 run identical active-token counts,
    so any kernel row-count effect applies to both in the same way: cuDNN attention at full
    resolution under PyTorch's default, cuBLAS at the small test sizes.
- **Attention** (registered, `density/attention.py`): generation on CUDA uses flash
  attention. PyTorch's own choice on the H100, cuDNN attention, was not reproducible across
  processes and gave different results for different query counts. Session D re-measured
  generation with flash on a fresh H100: every comparison is bitwise, across processes and
  across machines (pod D against pod B's flash diagnostic). Flash costs 27 percent more time
  than cuDNN for dense C1 and 22-23 percent for C3 and C4.
- **fp32 convolutions** (registered from the smoke test's comparison runs, by Chase's rule:
  strict fp32 unless it adds more than about 10 percent per video or per reference):
  - **VAE** (`density/wan.py`): TF32, cuDNN's default. Strict fp32 added 13.1 s per video
    and needed 41 GiB instead of 8.7 GiB.
  - **SAM 2.1** (`density/masks/registered.py`): strict fp32. On the slice it changed the
    per-reference time by less than 1 percent.
- **Hour caps:** each step checks the pod's age before it starts, and the session stops to
  ask before passing 6 hours on the slice or 4 hours on the H100.
- **Scripts** (`scripts/pod/`):
  - `sync.sh` (Mac): `prepare`, `code` (tracked files, `.git` and
    `results/checks/wan_real_mps/`; it refuses a dirty tree), `mask-weights`, `clips`,
    `pod-env`, `run`, `status` and `results`.
  - `bootstrap.sh`:
    - Installs uv 0.9.7, CPython 3.10.19 and the CUDA 13.0.3 toolkit, each verified
      against `configs/pod_installs.json`.
    - Builds the environment with `uv sync --locked` in two passes, torch first, then the
      sam-2 and Detectron2 builds.
    - Cleans the uv cache if uv can't hardlink.
    - Downloads Wan2.1 one pattern at a time and checks Detectron2's sm_90 and sm_120 code.
  - `session.sh`: runs a session in tmux and stops at the first failure.
  - `diagnostics/`: two follow-up runs from Session B on the H100. One compares HSA and dense
    module by module at step 0 and tests each attention backend. The other tests whether
    dense C1 is bitwise across processes.
  - `env.sh`: the environment every pod command sources. It sizes the CPU thread pools
    (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) to the pod's cgroup CPU quota. Pods see every
    host core, 128 on the slice, but the slice's quota is about 3.4 CPUs. With torch's
    default of 64 threads, CPU work thrashed under throttling: the tiny-model Milestone 1
    check took 19 minutes.
- **The volume** is MooseFS. `df` reports the whole storage cluster, so disk use is measured
  per folder with `du`. During uv's parallel unpacking the volume once returned a transient
  "Stale file handle". `bootstrap.sh` retries `uv sync` for that error only. uv can hardlink
  between its cache and the venv on the volume, so the cache is kept.
- **Pod log:** pod creation and deletion times (UTC), prices and billing go in
  `results/checks/smoke_test/pods.json`, written on the Mac from the RunPod MCP.
- **Dry run:** `--dry-run` runs every step on the Mac with tiny random models, writing to a
  folder given by `--out`. Use it before any pod time.

#### Results (2026-10-06, `results/checks/smoke_test.md`)

- **Status:** no open errors, stops, pilot blockers or bugs. Session B's "all HSA groups
  dense" miss (25.8 dB on smoke-2) came from cuDNN attention across processes; with flash it
  is bitwise.
- **Exactness with flash (Session D):** dense repeats, HSA with every group at full steps
  against dense, and T3's fast path against the oracle for C3 and C4 are all bitwise at full
  resolution. At 416 x 240, Milestone 2's T3 rows still differ from iteration 5, from cuBLAS
  at small row counts; they are recorded as measurements.
- **Time on the H100 with flash:** RE time 73.8 s. C3 takes 39.4 s and C4 39.6 s end to end,
  so C4's overhead is +0.19 s.
- **Memory:** peaks of 16.0 GiB in HSA denoising (18.6 GiB reserved) and 11.4 GiB in decode,
  so a 24 GB GPU fits.
- **Projections:** generation needs about 1.0 GPU-hour ($3.4-3.6) for the pilot and
  10.2-10.6 GPU-hours ($36-37) for the main run. Masks need 2.2 slice-hours ($1.28) for the
  pilot and 20.6 ($12.1) for the main run.
- **Masks on the slice (Session C):** 99 s per face reference and 551 s per manipulation
  reference, repeats identical. The slice was slower than the Mac's CPU on every stock clip;
  Session E found why (below).
- **Mask speed diagnosis (Session E, 2026-10-07):**
  - **SCRFD:** onnxruntime sized its pool to the host's 128 cores. With 141 OS threads on
    the 3.4-CPU quota, the cgroup throttled every period, and SCRFD took 0.78 s a frame.
    Capped at 3 threads it takes 0.067 s.
  - **Mask R-CNN and Hands23:** about 20 s an image at any thread count. With deterministic
    algorithms on and CUDA input, torchvision 0.29.1 replaces its roi_align kernel with a
    pure-Python version, lazily `torch.compile`d, that exists for a deterministic backward
    pass. Through the kernel, for inference only, they take 0.08 and 0.15 s. Repeats stay
    bitwise, and outputs differ from the Python version only in rounding (boxes by at most
    0.00012 px). The smoke-1 and smoke-2 masks were bitwise identical either way.
  - **Outputs on the slice:** every detection is bitwise identical across 1, 3 and 4
    threads and Session C's defaults. The smoke-1 and smoke-2 masks are identical at the
    defaults and at 3 threads; the other counts and the body clip weren't run through the
    pipeline. Strict deterministic mode runs the mask pipeline without stopping.
  - **The Mac's CPU (validated configuration: 8 torch threads):** 58.5 s per face
    reference, 2,077 s per manipulation reference and 502 s per body reference. On the Mac,
    SCRFD's results change with onnxruntime's thread count, and SAM's with torch's.
    ultralytics sets `OMP_NUM_THREADS=1` on import when it is unset, so the Mac's thread
    count must be pinned explicitly.
  - **du:** about 7-10 s over `/workspace` (57.7 GB allocated, 48.3 GB apparent).
- **Mask-check rerun (Session F, 2026-10-07)**, with the thread cap and the ROIAlign kernel:
  - 60/61 judged rows passed in 9.1 min (48 in Session C). The CUDA repeats are identical,
    and the WiLoR row is the known cross-device comparison.
  - The body clip's body path took 93 s (1,708 s in Session C).
  - The mask canary was recorded only after this validated, then passed from a fresh process
    in 18 s.
- **Spend:** six pods, 6.42 hours, $10.04 (H100 2.16 h, $7.52; slice 4.26 h, $2.51).

### Pilot guards (registered by Chase on 2026-10-07)

- **Tolerances** (`density/tolerances.py`): bitwise for the four full-resolution checks: the
  repeated dense run, all HSA groups dense against the dense loop, and the fast path against
  the masked-dense oracle for C3 and C4.
  - The scope is what Session D measured: 832 x 480, 81 frames, the H100 with flash
    attention, runs on the same device.
  - Smaller sizes are not covered: there cuBLAS gives different results for different row
    counts. Neither are comparisons across devices, which are reported, not judged.
  - Section 11's LPIPS check is met at zero for the registered setup, because the fast
    path's frames equal the oracle's exactly. LPIPS of Session B's cuDNN frames is skipped.
- **Canary** (`python -m density.canary`, `density/canary.py`): every H100 session starts
  by regenerating smoke-1's dense C1.
  - Its final latents and decoded uint8 frames must match pod D's hashes bitwise. If either
    differs, the session stops and reports this pod's host and driver beside pod D's.
  - Experiment jobs call `density.canary.require_passed()`. It accepts only a passing record
    from this host at the current, clean code.
  - Decoded frames have never been compared across hosts, so the first pilot pod is also the
    first cross-host test of the decode.
- **Strict determinism** (`determinism.configure(strict=True)`): in the pilot and the main
  run, an operation without a deterministic implementation, or any warning about
  deterministic algorithms, stops the job instead of being logged. The smoke checks keep
  warn-only mode and record.
- **Volume stop** (`density/volume.py`): `df` reports the whole storage cluster, so the volume is measured with `du`
  over `/workspace`. Runners stop cleanly, before starting a step or job, at 112 GB, read as 112 x 10^9 bytes
  (registered by Chase on 2026-10-07 for the 125 GB volume; it was 90 GB on 100 GB). The stop uses du's allocated size
  (`du -sk`), the larger figure: 57.7 GB against 48.3 GB of apparent size on 2026-10-07.
  - The smoke runner runs du before each step.
  - An experiment session (`volume.Room`, registered by Chase on 2026-10-07) runs one full du at its start. After each
    job it adds the allocated size (st_blocks) of every file the job wrote to a running total. It runs a full du again
    every 30 minutes, and whenever the running total reaches the stop, in which case that du decides.
  - A du took 52.5 s on the pilot's first H100 pod and 8-9 s on the slice.
- **C2 in the pilot:** 20 uniform dense steps on the 12 holdout prompts, 6 RE, as the HSA
  reproduction check. It is scored by full-frame LPIPS against the 40-step reference, the
  metric the HSA paper reports. HSA-50's median over the 12 prompts should be at most about 70
  percent of C2's median; if not, stop and investigate before the main run. The planning
  total is now 558 RE.

### Mask guards (decided by Chase on 2026-10-07)

- **CPU thread pools** (`density/cpu_threads.py`, `scripts/pod/env.sh`): on a pod, every
  pool runs at the cgroup CPU quota rounded down, 3 on the slice, with one inter-op thread
  for torch and onnxruntime. This isn't a registered value, since no output changed on the
  slice. Without a quota, as on the Mac, nothing changes.
- **ROIAlign** (registered, `density/masks/roi_align.py` and `registered.py`): Detectron2's
  ROIAlign runs through torchvision's kernel for inference. Under deterministic algorithms,
  torchvision would otherwise substitute its Python version, which exists for a
  deterministic backward pass. Every Mask R-CNN and Hands23 call checks that the kernel ran,
  and the kernel refuses to run with gradients on. On CPU it is the kernel torchvision calls
  anyway, so the Mac's masks are unchanged.
- **Mask canary** (`python -m density.mask_canary`, `density/mask_canary.py`): every mask
  session starts with it. Every model runs on the first 9 frames of the stock clips: SCRFD,
  WiLoR, Hands23, Mask R-CNN through the kernel, and SAM on each clip.
  - Each model's output hashes must equal `configs/mask_canary.json`, recorded in Session F.
    If any differs, the session stops and reports this pod's host, GPU, driver and CPU
    beside the registered pod's.
  - Mask jobs call `density.mask_canary.require_passed()` before they start.
- **Mask time** (protocol section 12): about 1.2 slice-hours for the pilot and 7.7 for the
  main run.

## Earlier plan: allocation checks

Build one small checker. Save a pass/fail report for these rules:

1. Lambda zero gives the same allocation as HSA.
2. An empty region mask gives the same allocation as HSA.
3. Equal scores follow the registered token-ID tie rule.
4. HSA and Density use equal active-token counts at each step.
5. The primary schedule uses 5,242 high-group tokens and 27,518 low-group tokens. Its actual token-step fraction is about 0.496007.
6. A half-covered cell keeps a coverage value of 0.5 before the stated maximum operations.
7. Pixel frame 0 maps separately. Pixel frames 1 through 4 map to the next latent frame.
8. The 3 by 3 maximum operation expands the mask through space only.

Use synthetic inputs first. Then test one short existing video and inspect the mask overlay.

## Current status

The Euler sampler, the HSA mechanics, the allocation rules, and the mask pipeline (on stock clips) are built and checked, and the GPU smoke test on RunPod is done (`results/checks/smoke_test.md`). Milestone 5 (the metrics, the prompt tooling, the runner with masks for Wan references, the pilot rules, the frozen analysis, the visual check and the registration builder) is built and dry-run on the Mac (`results/checks/metrics.md`). The pilot and an exploratory dose-response check have since run; the [README](../README.md) gives their results and the project's status.

Set up the Python 3.10 environment and run the checks:

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run python -m density.checks.euler
```

The last command writes `results/checks/euler_vs_diffusers.md` and `.json`, with package versions and the exact schedules.

The real-model check needs the Transformer and VAE (about 6.2 GB) in `weights/`:

```bash
uv run hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --revision 0fad780a534b6463e45facd96134c9f345acfa5b --include "transformer/*" --local-dir weights/Wan2.1-T2V-1.3B-Diffusers
```

```bash
uv run hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --revision 0fad780a534b6463e45facd96134c9f345acfa5b --include "vae/*" --local-dir weights/Wan2.1-T2V-1.3B-Diffusers
```

```bash
uv run python -m density.checks.wan_real
```

It writes `results/checks/wan_real_mps.md` and `.json` (weight and input hashes, sigmas, timings) and saves frames, a GIF, a contact sheet, and the exact inputs under `results/checks/wan_real_mps/`.

The HSA check reuses those inputs and weights, verifies their hashes against `wan_real_mps.json`, and takes about 16 minutes (`--tiny-only` skips the real model):

```bash
uv run python -m density.checks.hsa
```

It writes `results/checks/hsa_mechanics.md` and `.json`.

The allocation check uses the same inputs and weights and takes about 13 minutes (`--tiny-only` skips the real model and the demo):

```bash
uv run python -m density.checks.allocation
```

It writes `results/checks/allocation_rules.md`, `.json` and `.sha256`; the demo's overlay image goes under `results/checks/allocation_demo/`.

Full-resolution model checks, CUDA behavior, memory, and timing belong to the GPU smoke test (`results/checks/smoke_test.md`). Keep those results separate from these Mac checks.
