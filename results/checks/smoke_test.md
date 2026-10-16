# GPU smoke test (protocol sections 8, 11, 12 and 13)

Generated 2026-10-07T04:14:59+00:00. No experiment data: the two smoke prompts are excluded from every prompt pool.

Code: `2036a769b6236caaf8a7a5cc0e22caed5333be29`, `7473210fe517c4358ae3cd4fa2668a2f0e527612`, `7b9b39b641b1957aa52ebc4aa65a66be258efdb5`, `806a9392fa3d1568ca64f77cf9d2f163d01a4741`, `dbb965ead8e1c6d5cfcc7de0baf3252ba56d3a70`, `e948123132f1553814423f581cb77433c33362c5`, `eb8fb132e3a7fc3bf51192d99d657b5f9c5b8dc0`. Steps done: D:crossmachine, D:encode, D:env_D, D:generate, D:hsa_dense, D:milestones, D:noise, D:perf, D:sdpa, D:sheet, D:t3, D:weights_D, F:env_F, F:mask_check, F:mask_compare, F:weights_F, encode, env_A, env_B, env_C, env_E, generate, hsa_dense, mask_check, mask_compare, masks, milestones, noise, noise_mac, perf, pytest, sdpa, sheet, t3, vae_precision, weights_A, weights_B, weights_C, weights_E.

## Status

- **Resolved:** Session B (cuDNN, not registered): smoke-2: all HSA groups dense, decoded PSNR 25.79 dB is below 45 dB. Resolved in Session D: with flash attention registered, HSA with every group at full steps equals the dense C1 run from another process, bitwise.
- **Finding:** PyTorch chose cuDNN attention on the H100 at full resolution. It is bitwise within a process, but a dense C1 rerun in a new process differs from step 0 (smoke-2 max latent diff 3.76), and it is not row-invariant. The 'all HSA groups dense' miss compared two processes, so its cause is this, not HSA. With flash attention pinned, C1 is bitwise across processes, and HSA all-full vs dense (yes) and C3's fast path vs the oracle (yes) are bitwise; the 40 dense steps took 78 s with flash against 58 s with cuDNN (wall time after loading, with a copy to the host at every step). Session D measured flash's cost with CUDA events (end to end in steady state, flash against cuDNN): C1 73.8 s against 58.1 s (+27 percent), C3 39.4 s against 32.3 s (+22 percent), C4 39.6 s against 32.3 s (+23 percent). The projections use Session D's times.
- No open errors, stops, pilot blockers or bugs.
- **Registered:** Attention for generation on CUDA: FLASH_ATTENTION, then EFFICIENT_ATTENTION (density/attention.py); Session D re-measured with it.
- **Registered:** Starting noise: generated once on CPU, saved, and always loaded from the saved file by its SHA-256, never regenerated (density/noise.py); the same seed gives different noise on x86 and ARM CPUs.
- **Registered:** VAE convolutions: TF32, the CUDA default (density/wan.py): strict fp32 adds 13.1 to 13.2 s per video, at most 23 percent of C1's and 41 percent of C3's end-to-end time with Session B's times (18 and 33 percent with Session D's), over the 10 percent limit.
- **Registered:** SAM 2.1 convolutions, by the same rule: strict fp32 (IEEE) (density/masks/registered.py), since strict fp32 stays within 10 percent of per-reference mask time: -1.9 percent for smoke-1 (face), +0.8 percent for smoke-2 (manipulation).
- **Registered:** Tolerances for the four full-resolution checks: bitwise, with Session D's scope (density/tolerances.py); section 11's LPIPS check is met at zero.
- **Registered:** The H100 canary: smoke-1's dense C1 must match pod D's latent and frame hashes at the start of every H100 session (density/canary.py).
- **Registered:** Strict deterministic mode for the pilot and main run: a deterministic-algorithm warning stops the job (density/determinism.py).
- **Registered:** The volume stop: du's allocated size over /workspace before each step or batch, stopping at 90 x 10^9 bytes (density/volume.py).
- **Registered:** Detectron2's ROIAlign through torchvision's kernel for inference, checked on every call (density/masks/roi_align.py).
- **Registered:** The mask canary: every mask model on the stock clips' first 9 frames must hash as recorded in Session F (density/mask_canary.py, configs/mask_canary.json).
- Note: On pods the CPU pools run at the cgroup quota rounded down, with one inter-op thread (density/cpu_threads.py, scripts/pod/env.sh). Chase approved it on 2026-10-07; it is not a registered value, since no output changed on the slice.
- Note: Deterministic mode fills newly allocated memory (torch.utils.deterministic.fill_uninitialized_memory), which costs a little time in every run; the timings include it.

## Pods and spend

| Session | GPU | Pod | Created (UTC) | Deleted (UTC) | Hours | $/hr | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | umaruiv0mb0art | 2026-10-06T19:46:02.747000+00:00 | 2026-10-06T20:41:38+00:00 | 0.926 | 0.59 | 0.547 |
| B | NVIDIA H100 80GB HBM3 | efpu6nbag01t04 | 2026-10-06T20:52:26.133000+00:00 | 2026-10-06T22:16:57+00:00 | 1.41 | 3.49 | 4.92 |
| C | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | g3vp3gyin364hm | 2026-10-06T22:25:03.655000+00:00 | 2026-10-06T23:49:09+00:00 | 1.4 | 0.59 | 0.827 |
| D | NVIDIA H100 80GB HBM3 | 40skcaoggtjvcf | 2026-10-06T23:50:43.071000+00:00 | 2026-10-07T00:35:32+00:00 | 0.747 | 3.49 | 2.61 |
| E | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | iovy32wx4dzat6 | 2026-10-07T01:40:48.969000+00:00 | 2026-10-07T03:22:08+00:00 | 1.69 | 0.59 | 0.996 |
| F | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | pj1h2kouyzvs1a | 2026-10-07T03:57:03.063000+00:00 | 2026-10-07T04:11:46+00:00 | 0.245 | 0.59 | 0.145 |

| GPU | Pods | Hours | Cost ($) |
| --- | --- | --- | --- |
| NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | 4 | 4.26 | 2.51 |
| NVIDIA H100 80GB HBM3 | 2 | 2.16 | 7.52 |

Total pod spend: $10.04. Hours come from the creation and deletion times logged on the Mac.

RunPod's billing records, queried 2026-10-07T03:23:00+00:00: $9.29 posted (pod A $0.55, pod B $4.91, pod C $0.83, pod D $2.60, pod E $0.40, pod F not posted yet). Billing lags: pod E's record (deleted 03:22 UTC) is still posting, so the tables use the logged creation and deletion times; per-pod amounts sum RunPod's daily records.

## Pod environments

|  | A | B | C | E | D | F |
| --- | --- | --- | --- | --- | --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (sm 12.0, 23.6 GiB) | NVIDIA H100 80GB HBM3 (sm 9.0, 79.2 GiB) | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (sm 12.0, 23.6 GiB) | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (sm 12.0, 23.6 GiB) | NVIDIA H100 80GB HBM3 (sm 9.0, 79.2 GiB) | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (sm 12.0, 23.6 GiB) |
| Driver | NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.16, [Insufficient Permissions], 12.0, Enabled | NVIDIA H100 80GB HBM3, 580.126.09, 81559 MiB, 9.0, Disabled | NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.16, [Insufficient Permissions], 12.0, Enabled | NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.16, [Insufficient Permissions], 12.0, Enabled | NVIDIA H100 80GB HBM3, 580.126.09, 81559 MiB, 9.0, Disabled | NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.16, [Insufficient Permissions], 12.0, Enabled |
| Driver CUDA | 13.0 | 13.0 | 13.0 | 13.0 | 13.0 | 13.0 |
| torch / CUDA / cuDNN | 2.14.1+cu130 / 13.0 / 92400 | 2.14.1+cu130 / 13.0 / 92400 | 2.14.1+cu130 / 13.0 / 92400 | 2.14.1+cu130 / 13.0 / 92400 | 2.14.1+cu130 / 13.0 / 92400 | 2.14.1+cu130 / 13.0 / 92400 |
| CUDA toolkit | Cuda compilation tools, release 13.0, V13.0.88 | Cuda compilation tools, release 13.0, V13.0.88 | Cuda compilation tools, release 13.0, V13.0.88 | Cuda compilation tools, release 13.0, V13.0.88 | Cuda compilation tools, release 13.0, V13.0.88 | Cuda compilation tools, release 13.0, V13.0.88 |
| Python, OS | 3.10.19, Ubuntu 24.04.3 LTS, kernel 6.8.0-111-generic | 3.10.19, Ubuntu 24.04.3 LTS, kernel 6.8.0-90-generic | 3.10.19, Ubuntu 24.04.3 LTS, kernel 6.8.0-111-generic | 3.10.19, Ubuntu 24.04.3 LTS, kernel 6.8.0-111-generic | 3.10.19, Ubuntu 24.04.3 LTS, kernel 6.8.0-90-generic | 3.10.19, Ubuntu 24.04.3 LTS, kernel 6.8.0-111-generic |
| CPU, RAM | AMD EPYC 9355 32-Core Processor: 3.4 CPUs allowed of 128 on the host; 57.7 GiB RAM allowed | Intel(R) Xeon(R) Platinum 8470: 22.1 CPUs allowed of 208 on the host; 234 GiB RAM allowed | AMD EPYC 9355 32-Core Processor: 3.4 CPUs allowed of 128 on the host; 57.7 GiB RAM allowed | AMD EPYC 9355 32-Core Processor: 3.4 CPUs allowed of 128 on the host; 57.7 GiB RAM allowed | Intel(R) Xeon(R) Platinum 8470: 22.1 CPUs allowed of 208 on the host; 234 GiB RAM allowed | AMD EPYC 9355 32-Core Processor: 3.4 CPUs allowed of 128 on the host; 57.7 GiB RAM allowed |
| Code | 7473210fe517c4358ae3cd4fa2668a2f0e527612 (dirty: False) | 7b9b39b641b1957aa52ebc4aa65a66be258efdb5 (dirty: False) | dbb965ead8e1c6d5cfcc7de0baf3252ba56d3a70 (dirty: False) | 806a9392fa3d1568ca64f77cf9d2f163d01a4741 (dirty: False) | e948123132f1553814423f581cb77433c33362c5 (dirty: False) | 2036a769b6236caaf8a7a5cc0e22caed5333be29 (dirty: False) |
| Deterministic settings | CUBLAS_WORKSPACE_CONFIG :4096:8, cudnn_benchmark False, cudnn_deterministic False, deterministic_algorithms True, deterministic_warn_only True, fill_uninitialized_memory True, float32_matmul_precision highest, cuda_matmul_allow_tf32 False, cuda_matmul_fp32_precision none, cudnn_allow_tf32 True, cudnn_conv_fp32_precision tf32 | CUBLAS_WORKSPACE_CONFIG :4096:8, cudnn_benchmark False, cudnn_deterministic False, deterministic_algorithms True, deterministic_warn_only True, fill_uninitialized_memory True, float32_matmul_precision highest, cuda_matmul_allow_tf32 False, cuda_matmul_fp32_precision none, cudnn_allow_tf32 True, cudnn_conv_fp32_precision tf32 | CUBLAS_WORKSPACE_CONFIG :4096:8, cudnn_benchmark False, cudnn_deterministic False, deterministic_algorithms True, deterministic_warn_only True, fill_uninitialized_memory True, float32_matmul_precision highest, cuda_matmul_allow_tf32 False, cuda_matmul_fp32_precision none, cudnn_allow_tf32 True, cudnn_conv_fp32_precision tf32 | CUBLAS_WORKSPACE_CONFIG :4096:8, cudnn_benchmark False, cudnn_deterministic False, deterministic_algorithms True, deterministic_warn_only True, deterministic_warnings_are_errors False, fill_uninitialized_memory True, float32_matmul_precision highest, cuda_matmul_allow_tf32 False, cuda_matmul_fp32_precision none, cudnn_allow_tf32 True, cudnn_conv_fp32_precision tf32 | CUBLAS_WORKSPACE_CONFIG :4096:8, cudnn_benchmark False, cudnn_deterministic False, deterministic_algorithms True, deterministic_warn_only True, fill_uninitialized_memory True, float32_matmul_precision highest, cuda_matmul_allow_tf32 False, cuda_matmul_fp32_precision none, cudnn_allow_tf32 True, cudnn_conv_fp32_precision tf32 | CUBLAS_WORKSPACE_CONFIG :4096:8, cudnn_benchmark False, cudnn_deterministic False, deterministic_algorithms True, deterministic_warn_only True, deterministic_warnings_are_errors False, fill_uninitialized_memory True, float32_matmul_precision highest, cuda_matmul_allow_tf32 False, cuda_matmul_fp32_precision none, cudnn_allow_tf32 True, cudnn_conv_fp32_precision tf32, cpu_threads {'quota_threads': 3, 'torch_intra_op': 3, 'torch_inter_op': 1} |
| SDPA backends enabled | flash, mem_efficient, math, cudnn | flash, mem_efficient, math, cudnn | flash, mem_efficient, math, cudnn | flash, mem_efficient, math, cudnn | flash, mem_efficient, math, cudnn | flash, mem_efficient, math, cudnn |
| Detectron2 archs | sm_120, sm_90 | sm_120, sm_90 | sm_120, sm_90 | sm_120, sm_90 | sm_120, sm_90 | sm_120, sm_90 |
| sam-2 CUDA extension | no | no | no | no | no | no |
| /workspace df (the whole network-storage cluster, not the 100 GB volume) | 677811.7 of 1015682.3 GiB used | 678594.6 of 1015682.3 GiB used | 679020.7 of 1015682.3 GiB used | 680760.2 of 1015682.3 GiB used | 679046.7 of 1015682.3 GiB used | 682148.6 of 1015682.3 GiB used |

## Session B (H100): PyTorch's default attention, cuDNN at full resolution

### Tests and Milestones 1-3 on CUDA

- pytest: 109 passed, 24 skipped in 295.09s (0:04:55)
- M1: 20/20 judged rows passed; 0 same-device numeric differences (measurements); 0 failures; 4 unjudged.
- M2: 42/51 judged rows passed; 9 same-device numeric differences (measurements); 0 failures; 0 unjudged.
  - measurement: T3 fast path vs oracle (tiny model, fp32, seed 0): max abs diff 1.91e-06 from step 5
  - measurement: T3 fast path vs oracle (tiny model, fp32, seed 1): max abs diff 1.58e-06 from step 5
  - measurement: T3 fast path vs oracle (tiny model, fp32, seed 2): max abs diff 1.79e-06 from step 5
  - measurement: T3 fast path vs oracle (tiny model, bf16, seed 0): max abs diff 0.0325 from step 6
  - measurement: T3 fast path vs oracle (tiny model, bf16, seed 1): max abs diff 0.0299 from step 11
  - measurement: T3 fast path vs oracle (tiny model, bf16, seed 2): max abs diff 0.0306 from step 9
  - measurement: T3 fast path vs oracle (real model, bf16, seed 0): max abs diff 0.242 from step 5
  - measurement: T3 fast path vs oracle (real model, bf16, seed 1): max abs diff 0.282 from step 5
  - measurement: T3 fast path vs oracle (real model, bf16, seed 2): max abs diff 0.262 from step 5
- M3: 58/58 judged rows passed; 0 same-device numeric differences (measurements); 0 failures; 2 unjudged.
- T1's frame-hash row and allocation's R1-frames row compare against hashes recorded on MPS; on another device they are reported, not judged.

### Dense C1 at full resolution, twice per prompt

| Prompt | Bitwise | Max abs latent diff | First divergent step | Decoded PSNR (dB) | uint8 identical | VAE repeat max diff | Denoise (s) | Decode (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | yes | 0 |  | inf | yes | 0 | 54.9 | 3.27 |
| smoke-2 | yes | 0 |  | inf | yes | 0 | 54.8 | 3.28 |

### All HSA groups dense vs dense, full resolution

| Prompt | Bitwise | Max abs latent diff | First divergent step | Decoded PSNR (dB) | Relative to repeat noise |
| --- | --- | --- | --- | --- | --- |
| smoke-1 | no | 0.368 | 0 | 48.1 | the repeat runs were bitwise identical, so the noise floor is zero and the ratio is undefined |
| smoke-2 | no | 3.76 | 0 | 25.8 | the repeat runs were bitwise identical, so the noise floor is zero and the ratio is undefined |

### T3: fast path vs masked-dense oracle, full resolution, HSA-50

C4 at lambda 3 with a centered disk of radius 120 px (5796 tokens with coverage). Each oracle uses the labels its fast-path run assigned.

| Prompt | Condition | Bitwise | Max abs latent diff | First divergent step | Decoded PSNR (dB) | PSNR of the saved 8-bit frames (dB) | Token-step fraction | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | C3 | no | 0.599 | 5 | 46.7 | 46.2 | 0.496 | measurement: on identical inputs, blocks.0.attn1 gives different results for different row counts (attention query-count dependence) |
| smoke-1 | C4 | no | 0.633 | 5 | 46.1 | 45.7 | 0.496 | measurement: on identical inputs, blocks.0.attn1 gives different results for different row counts (attention query-count dependence) |
| smoke-2 | C3 | no | 1.78 | 5 | 32.9 | 32.9 | 0.496 | measurement: on identical inputs, blocks.0.attn1 gives different results for different row counts (attention query-count dependence) |
| smoke-2 | C4 | no | 3.81 | 5 | 26.5 | 26.5 | 0.496 | measurement: on identical inputs, blocks.0.attn1 gives different results for different row counts (attention query-count dependence) |

Dense repeats within one process are bitwise on CUDA, so repeat-run noise is zero and is no yardstick for T3: the deviation is reported as decoded PSNR. The structural checks are the bug test. (Across processes, cuDNN attention does not reproduce: see the diagnostics.)
C3 and C4 run identical active-token counts at every iteration, layer and guidance branch, so every kernel receives the same row counts in both conditions and any row-count effect applies to both in the same way. At full resolution on the H100 that effect comes from cuDNN attention (the GEMMs were row-invariant); in the tiny-model rehearsal on the slice it came from cuBLAS. Checked from the run logs: smoke-1 yes, smoke-2 yes.
Lossless frames for LPIPS: Not measured yet: no LPIPS model is pinned. The fast-path and oracle frames for C3 and C4 are saved losslessly in smoke_test/t3_frames/. In Milestone 5, compute their LPIPS and compare it with the improvement the experiment seeks (protocol sections 9 and 11): GO needs a median relative improvement of at least 10 percent of HSA's loss, and the fidelity guardrails allow at most 0.01 LPIPS degradation.
Fast-path repeat (C3 twice): smoke-1: bitwise yes, max abs diff 0; smoke-2: bitwise yes, max abs diff 0

- Evidence, smoke-1 C3, step 5: first differing module `blocks.0.attn1` (cond); inputs equal: yes; kernels alone reproduce it: yes; replay matches the run: yes.
- Evidence, smoke-1 C4, step 5: first differing module `blocks.0.attn1` (cond); inputs equal: yes; kernels alone reproduce it: yes; replay matches the run: yes.
- Evidence, smoke-2 C3, step 5: first differing module `blocks.0.attn1` (cond); inputs equal: yes; kernels alone reproduce it: yes; replay matches the run: yes.
- Evidence, smoke-2 C4, step 5: first differing module `blocks.0.attn1` (cond); inputs equal: yes; kernels alone reproduce it: yes; replay matches the run: yes.

### Attention backend (SDPA)

- dense: CUDNN_ATTENTION for query [1, 12, 32760, 128] x key [1, 12, 32760, 128] bfloat16 (60 calls); CUDNN_ATTENTION for query [1, 12, 32760, 128] x key [1, 12, 512, 128] bfloat16 (60 calls); profiler kernels: ['flash', 'cudnn']
- hsa_subset: CUDNN_ATTENTION for query [1, 12, 5242, 128] x key [1, 12, 32760, 128] bfloat16 (60 calls); CUDNN_ATTENTION for query [1, 12, 5242, 128] x key [1, 12, 512, 128] bfloat16 (60 calls); profiler kernels: ['flash', 'cudnn']
- Same backend for dense and subset query counts: yes.

### Memory and timing (CUDA events)

Prompt smoke-1; 2 dense steps, 6 HSA-50 steps and one decode, discarded; T5 never loaded; KV caches released before each decode.

| Condition | Token-step fraction | Denoise (s) | Spread (s) | Decode (s) | End to end (s) | Peak alloc denoise (GiB) | Peak reserved denoise (GiB) | Peak alloc decode (GiB) | Peak reserved decode (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 1 | 54.9 | 0.0946 | 3.27 | 58.1 | 4.72 | 5.6 | 11.4 | 16 |
| C3 | 0.496 | 31.6 | 5.19 | 3.28 | 34.9 | 16 | 18.6 | 11.4 | 15.1 |
| C4 | 0.496 | 29 | 0.0161 | 3.27 | 32.3 | 16 | 18.6 | 11.4 | 15.1 |

C4's overhead over C3 in steady state (best of two runs each): -0.01 s (-0.05 percent). First run in excess of the best one: C1 +0.1 s, C3 +5.2 s, C4 +0.0 s. The means above include that first-call cost.
RE time: 58.1 s (one dense 40-step generation (protocol section 12): CUDA-event denoising plus VAE decoding, with precomputed text embeddings). Weights resident after loading: 3.16 GiB.

### TF32 vs IEEE fp32 convolutions

Comparison runs for the registered choices: TF32 for the VAE (density/wan.py) and strict fp32 for SAM 2.1 (density/masks/registered.py), by the rule below.

| Prompt | VAE decode, TF32 (s) | VAE decode, IEEE (s) | Decoded PSNR, IEEE vs TF32 (dB) | Max abs diff | uint8 identical |
| --- | --- | --- | --- | --- | --- |
| smoke-1 | 3.28 | 16.4 | 85.6 | 0.00409 | no |
| smoke-2 | 3.28 | 16.4 | 81 | 0.00694 | no |

Rule (Chase): strict fp32 unless it adds more than about 10% to per-video time.
- VAE: IEEE fp32 adds 13.1 to 13.2 s per video; the larger is 22.6 percent of C1's and 40.7 percent of C3's end-to-end time. Within 10 percent: no.
- Masks, smoke-1 (face): IEEE fp32 takes 96.8 s against 98.7 s (-1.9 percent); identical masks: no. Within 10 percent: yes.
- Masks, smoke-2 (manipulation): IEEE fp32 takes 520.1 s against 515.9 s (+0.8 percent); identical masks: no. Within 10 percent: yes.
- Masks, smoke-1: IEEE fp32 convolutions change the masks: {"F": {"pixels_differing": 538, "frames_differing": 81, "iou": 0.9999383926693232}}; 96.8 s with the models already built.
- Masks, smoke-2: IEEE fp32 convolutions change the masks: {"C": {"pixels_differing": 334, "frames_differing": 34, "iou": 0.9998540002404188}, "H": {"pixels_differing": 1413, "frames_differing": 73, "iou": 0.999717930403669}, "O": {"pixels_differing": 27, "frames_differing": 22, "iou": 0.9999918291678733}}; 520.1 s with the models already built.

### References and frame 0

Contact sheet `results/checks/smoke_test/references/contact_sheet.png` (sha256 `a8eb6f4db9632ce8...`), frames [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80].

| Prompt | Frame 0 luma | Frames 1-4 mean | Other frames mean (std) | Other frames max | Brighter than the mean | Brighter than every other frame |
| --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | 0.355 | 0.356 | 0.3732 (0.0236) | 0.43 | no | no |
| smoke-2 | 0.488 | 0.492 | 0.4734 (0.0128) | 0.493 | yes | no |

## Session D (H100): flash attention, registered

### Tests and Milestones 1-3 on CUDA

- M1: 20/20 judged rows passed; 0 same-device numeric differences (measurements); 0 failures; 4 unjudged.
- M2: 45/51 judged rows passed; 6 same-device numeric differences (measurements); 0 failures; 0 unjudged.
  - measurement: T3 fast path vs oracle (tiny model, fp32, seed 0): max abs diff 1.91e-06 from step 5
  - measurement: T3 fast path vs oracle (tiny model, fp32, seed 1): max abs diff 1.58e-06 from step 5
  - measurement: T3 fast path vs oracle (tiny model, fp32, seed 2): max abs diff 1.79e-06 from step 5
  - measurement: T3 fast path vs oracle (real model, bf16, seed 0): max abs diff 0.179 from step 5
  - measurement: T3 fast path vs oracle (real model, bf16, seed 1): max abs diff 0.249 from step 5
  - measurement: T3 fast path vs oracle (real model, bf16, seed 2): max abs diff 0.26 from step 5
- M3: 58/58 judged rows passed; 0 same-device numeric differences (measurements); 0 failures; 2 unjudged.
- T1's frame-hash row and allocation's R1-frames row compare against hashes recorded on MPS; on another device they are reported, not judged.

### Dense C1 at full resolution, twice per prompt

| Prompt | Bitwise | Max abs latent diff | First divergent step | Decoded PSNR (dB) | uint8 identical | VAE repeat max diff | Denoise (s) | Decode (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | yes | 0 |  | inf | yes | 0 | 70.7 | 3.11 |
| smoke-2 | yes | 0 |  | inf | yes | 0 | 70.8 | 3.11 |

### All HSA groups dense vs dense, full resolution

| Prompt | Bitwise | Max abs latent diff | First divergent step | Decoded PSNR (dB) | Relative to repeat noise |
| --- | --- | --- | --- | --- | --- |
| smoke-1 | yes | 0 |  | inf | 0 |
| smoke-2 | yes | 0 |  | inf | 0 |

### T3: fast path vs masked-dense oracle, full resolution, HSA-50

C4 at lambda 3 with a centered disk of radius 120 px (5796 tokens with coverage). Each oracle uses the labels its fast-path run assigned.

| Prompt | Condition | Bitwise | Max abs latent diff | First divergent step | Decoded PSNR (dB) | PSNR of the saved 8-bit frames (dB) | Token-step fraction | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | C3 | yes | 0 |  | inf | inf | 0.496 | bitwise |
| smoke-1 | C4 | yes | 0 |  | inf | inf | 0.496 | bitwise |
| smoke-2 | C3 | yes | 0 |  | inf | inf | 0.496 | bitwise |
| smoke-2 | C4 | yes | 0 |  | inf | inf | 0.496 | bitwise |

The fast path equals the oracle bitwise at full resolution for C3 and C4. C3 and C4 run identical active-token counts at every iteration, layer and guidance branch (checked from the run logs: smoke-1 yes, smoke-2 yes).
Fast-path repeat (C3 twice): smoke-1: bitwise yes, max abs diff 0; smoke-2: bitwise yes, max abs diff 0


### Attention backend (SDPA)

- dense: FLASH_ATTENTION for query [1, 12, 32760, 128] x key [1, 12, 32760, 128] bfloat16 (60 calls); FLASH_ATTENTION for query [1, 12, 32760, 128] x key [1, 12, 512, 128] bfloat16 (60 calls); profiler kernels: ['flash']
- hsa_subset: FLASH_ATTENTION for query [1, 12, 5242, 128] x key [1, 12, 32760, 128] bfloat16 (60 calls); FLASH_ATTENTION for query [1, 12, 5242, 128] x key [1, 12, 512, 128] bfloat16 (60 calls); profiler kernels: ['flash']
- Same backend for dense and subset query counts: yes.

### Memory and timing (CUDA events)

Prompt smoke-1; 2 dense steps, 6 HSA-50 steps and one decode, discarded; T5 never loaded; KV caches released before each decode.

| Condition | Token-step fraction | Denoise (s) | Spread (s) | Decode (s) | End to end (s) | Peak alloc denoise (GiB) | Peak reserved denoise (GiB) | Peak alloc decode (GiB) | Peak reserved decode (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 1 | 70.7 | 0.104 | 3.11 | 73.8 | 4.72 | 5.6 | 11.4 | 16 |
| C3 | 0.496 | 36.4 | 0.18 | 3.11 | 39.5 | 16 | 18.6 | 11.4 | 15.1 |
| C4 | 0.496 | 36.5 | 0.0123 | 3.11 | 39.6 | 16 | 18.6 | 11.4 | 15.1 |

C4's overhead over C3 in steady state (best of two runs each): +0.19 s (+0.53 percent). First run in excess of the best one: C1 +0.1 s, C3 +0.2 s, C4 +0.0 s. The means above include that first-call cost.
RE time: 73.8 s (one dense 40-step generation (protocol section 12): CUDA-event denoising plus VAE decoding, with precomputed text embeddings). Weights resident after loading: 3.16 GiB.

### References and frame 0

Contact sheet `results/checks/smoke_test/flash/references/contact_sheet.png` (sha256 `5377c7b85a7006ea...`), frames [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80].

| Prompt | Frame 0 luma | Frames 1-4 mean | Other frames mean (std) | Other frames max | Brighter than the mean | Brighter than every other frame |
| --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | 0.354 | 0.356 | 0.3729 (0.0235) | 0.429 | no | no |
| smoke-2 | 0.488 | 0.49 | 0.4732 (0.0127) | 0.495 | yes | no |

### Across machines: Session D against pod B's flash diagnostic

- pod B: 9fe63ec98193, NVIDIA H100 80GB HBM3, 580.126.09, 81559 MiB, 9.0, Disabled, driver CUDA 13.0, Intel(R) Xeon(R) Platinum 8470, kernel 6.8.0-90-generic.
- this pod: 0d7e358ab3af, NVIDIA H100 80GB HBM3, 580.126.09, 81559 MiB, 9.0, Disabled, driver CUDA 13.0, Intel(R) Xeon(R) Platinum 8470, kernel 6.8.0-90-generic.
- dense_smoke_2_flash: {"bitwise_equal": true, "max_abs_diff": 0.0, "first_divergent_step": null, "pod_b_file": "/workspace/smoke_scratch/dense_smoke-2_flash.safetensors", "pod_b_trajectory_sha256": "6cf8a5b3d5bd4f99c2bf4cfa9f09d3ebeb15d828888be5e45b1aec892502784c"}
- c3_labels_smoke_2: {"identical": true, "this_pod": "6e17a4c39319fec87df37600399cdf59785092c4d2bc90d31ac5503154e76953", "pod_b": "6e17a4c39319fec87df37600399cdf59785092c4d2bc90d31ac5503154e76953"}

## Tolerance calibration (protocol section 11)

- **Repeated dense run (C1 twice, same device)** (protocol: decoded PSNR at least 45 dB; record tensor errors and backend determinism). Observed: {"bitwise": true, "max_abs_latent_diff": 0.0, "min_decoded_psnr_db": Infinity}. **Registered** (2026-10-07, Chase, density/tolerances.py): bitwise. Scope: 832 x 480, 81 frames, on the H100 with flash attention registered (Session D), same device. It does not cover smaller sizes, where Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still differ from iteration 5 (cuBLAS at small row counts), or results across devices, which are reported, not judged.
- **All HSA groups dense vs the dense loop** (protocol: match the same dense Euler path; decoded PSNR at least 45 dB). Observed: {"bitwise": true, "max_abs_latent_diff": 0.0, "min_decoded_psnr_db": Infinity}. **Registered** (2026-10-07, Chase, density/tolerances.py): bitwise. Scope: 832 x 480, 81 frames, on the H100 with flash attention registered (Session D), same device. It does not cover smaller sizes, where Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still differ from iteration 5 (cuBLAS at small row counts), or results across devices, which are reported, not judged.
- **Fast path vs masked-dense oracle at full resolution, C3 (T3)** (protocol: compare tensor error against the dense-path numerical baseline; check that its induced LPIPS difference is small compared with the improvement the experiment seeks; freeze the measured tolerance before the pilot). Observed: {"bitwise": true, "min_decoded_psnr_db": Infinity, "min_decoded_psnr_db_uint8": Infinity, "max_abs_latent_diff": 0.0, "verdicts": ["bitwise", "bitwise"]}. **Registered** (2026-10-07, Chase, density/tolerances.py): bitwise. Scope: 832 x 480, 81 frames, on the H100 with flash attention registered (Session D), same device. It does not cover smaller sizes, where Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still differ from iteration 5 (cuBLAS at small row counts), or results across devices, which are reported, not judged.
- **Fast path vs masked-dense oracle at full resolution, C4 (T3)** (protocol: compare tensor error against the dense-path numerical baseline; check that its induced LPIPS difference is small compared with the improvement the experiment seeks; freeze the measured tolerance before the pilot). Observed: {"bitwise": true, "min_decoded_psnr_db": Infinity, "min_decoded_psnr_db_uint8": Infinity, "max_abs_latent_diff": 0.0, "verdicts": ["bitwise", "bitwise"]}. **Registered** (2026-10-07, Chase, density/tolerances.py): bitwise. Scope: 832 x 480, 81 frames, on the H100 with flash attention registered (Session D), same device. It does not cover smaller sizes, where Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still differ from iteration 5 (cuBLAS at small row counts), or results across devices, which are reported, not judged.
- **All tokens active in cached attention (T2, small resolution, same device)** (protocol: match dense attention within the calibrated numerical tolerance; record relative error). Observed: {"measurements": [], "failures": []}.
- **T10 lambda = 0 equals C3 (small resolution, same device)** (protocol: identical group labels and active-token IDs; output within the baseline numerical tolerance). Observed: {"non_bitwise_rows": [], "failed_rows": []}.
- **T11 empty mask equals C3 (small resolution, same device)** (protocol: C4 reproduces C3; any unexplained difference is an implementation failure). Observed: {"non_bitwise_rows": [], "failed_rows": []}.
- LPIPS: With flash attention registered, the fast path's frames equal the oracle's exactly for C3 and C4 (Session D), so their LPIPS is zero by construction. Session B's frames under cuDNN, which differ, stay saved losslessly in smoke_test/t3_frames/. The section 11 check is met at zero for the registered configuration (Chase, 2026-10-07): with flash attention, the fast path's frames equal the oracle's exactly for C3 and C4, so their LPIPS is zero; LPIPS of Session B's cuDNN frames is skipped.

## Masks on the slice

| Reference | Stratum | Seconds | Repeat seconds | Repeat identical | Eligibility | Usable ROI frames | Components (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1 | face | 99.4 | 99.4 | yes | pass | 81 | total 99.4, detect_track 98.3, roi_coverage_measurement_eligibility 1.1, F.detector 82.0, F.sam 10.2 |
| smoke-2 | manipulation | 551 | 517 | yes | pass | 81 | total 550.5, detect_track 549.3, roi_coverage_measurement_eligibility 1.2, H.detector 3.6, H.sam 281.2, O.hands23 204.0, O.sam 53.9 |

SAM 2.1's attention (fp32): EFFICIENT_ATTENTION for all 255 calls of a 9-frame probe.

Stock-clip mask check on NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb: 60/61 judged rows passed in 48.4 min. The WiLoR conversion row compares against detections recorded on the Mac's ARM CPU, so on the pod's x86 CPU it is a cross-device comparison (reported, not judged); the conversion stays verified on the Mac.

Stock clips, slice vs the Mac's committed masks (reported, not judged):

| Clip | Regions | Tracks identical | Events identical | Decision (Mac / pod) | Seconds (Mac CPU / pod) |
| --- | --- | --- | --- | --- | --- |
| face | F differs (680 px, IoU 0.9999); ROI differs (680 px, IoU 0.9999) | no | yes | pass / pass | 59 / 100 |
| hands | H differs (23696 px, IoU 0.9935); O differs (41 px, IoU 1.0000); C differs (1116 px, IoU 0.9992); ROI differs (21862 px, IoU 0.9958) | no | yes | pass / pass | 175 / 228 |
| body | B differs (48 px, IoU 1.0000); H differs (2896 px, IoU 0.9541); O identical; C identical; ROI differs (48 px, IoU 1.0000) | no | no | not gated / not gated | 1756 / 2186 |
| landscape | ROI identical | yes | yes | not gated / not gated | 0 / 0 |

## Session F (slice): the mask check with the registered configuration

The stock-clip mask check rerun on the slice with the CPU pools at the quota (3 threads) and Detectron2's ROIAlign through torchvision's kernel (registered 2026-10-07), the body clip included; then the mask canary recorded and checked.

- Mask check: 60/61 judged rows passed in 9.1 min. T26 repeated CUDA run gives identical masks: face: yes; T26 repeated CUDA run gives identical masks: hands: yes; T26 SAM 2.1 on MPS: not run on this device. The WiLoR conversion row is the cross-device comparison against detections recorded on the Mac's ARM CPU (reported, not judged).

| Clip | Seconds | Components (s) |
| --- | --- | --- |
| face | 21.2 | detect_track 20, roi_coverage_measurement_eligibility 1, F.detector 6, F.sam 10 |
| hands | 38.0 | detect_track 37, roi_coverage_measurement_eligibility 1, H.detector 1, H.sam 9, O.hands23 2, O.sam 19 |
| body | 353 | detect_track 352, roi_coverage_measurement_eligibility 1, B.detector 7, B.sam 85, H.detector 1, H.sam 244, O.hands23 2, O.sam 9 |
| landscape | 0.0 | detect_track 0, roi_coverage_measurement_eligibility 0 |

The body clip's body path (B detector, B SAM and the ROI measures) took 93 s with the production code, against 94 s in Session E's follow-up.

Stock clips, slice vs the Mac's committed masks (reported, not judged):

| Clip | Regions | Tracks identical | Events identical | Decision (Mac / pod) | Seconds (Mac CPU / pod) |
| --- | --- | --- | --- | --- | --- |
| face | F differs (732 px, IoU 0.9999); ROI differs (732 px, IoU 0.9999) | no | yes | pass / pass | 59 / 21 |
| hands | H differs (23652 px, IoU 0.9935); O differs (14 px, IoU 1.0000); C differs (1057 px, IoU 0.9992); ROI differs (21833 px, IoU 0.9958) | no | yes | pass / pass | 175 / 38 |
| body | B differs (5 px, IoU 1.0000); H differs (2894 px, IoU 0.9541); O identical; C identical; ROI differs (5 px, IoU 1.0000) | no | no | not gated / not gated | 1756 / 353 |
| landscape | ROI identical | yes | yes | not gated / not gated | 0 / 0 |

- Gate before recording the canary: passed.
- Mask canary recorded on 04067d68a106 (NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, driver 580.126.16, CPU AMD EPYC 9355 32-Core Processor) at code 2036a76: 7 hashes (hands23, mask_rcnn, sam_body, sam_face, sam_hands, scrfd, wilor); counts {"scrfd": [1, 1, 1, 1, 1, 1, 1, 1, 1], "wilor": [1, 1, 1, 1, 1, 1, 1, 1, 1], "hands23": [1, 1], "mask_rcnn": [2, 2], "roi_align_kernel_calls": 40, "onnxruntime_threads": 3, "sam_face": 605869, "sam_hands": 404415, "sam_body": 203314}; 16 s.
- Mask canary checked from a fresh process: passed in 18 s.

## Projected GPU time and cost

RE time on the H100: 73.8 s. Price $3.49/hr.

| Phase | Planning RE | GPU-hours (RE x RE time) | $ (RE) | GPU-hours (by condition) | $ (by condition) |
| --- | --- | --- | --- | --- | --- |
| pilot | 54 | 1.11 | 3.86 | 1.15 | 4 |
| main | 498 | 10.2 | 35.6 | 10.6 | 37.1 |

Run times used: C1 73.8 s (measured); C2 38.4 s (estimated: half of C1's denoising time plus one decode); C3 39.4 s (measured); C4 39.6 s (measured); C5 39.6 s (assumed equal to C4: the same HSA machinery with a different ranking rule); C6 39.6 s (assumed equal to C4: the same HSA machinery with a different ranking rule); A1 39.6 s (assumed equal to C4: the same HSA machinery with a different ranking rule); C3 HSA-75B 56.6 s (estimated: denoising interpolated linearly in token-step fraction between C3 and C1); C4 HSA-75B 56.8 s (estimated as C3 HSA-75B plus C4's measured overhead); C3 switch 39.4 s (assumed equal to C3 (token-step fraction 0.5)); C4 switch 39.6 s (assumed equal to C4 (token-step fraction 0.5)).

| Masks | References | Slice hours | $ |
| --- | --- | --- | --- |
| pilot | face 12, manipulation 12 | 1.21 | 0.716 |
| main | face 72, manipulation 72, body 16, control 16 | 7.69 | 4.54 |

Seconds per reference: control 0.0 (stock landscape clip on the slice: the explicit empty mask); face 18.0 (smoke-1 on the slice with the registered configuration: CPU pools at the quota, the ROIAlign kernel (Session E's follow-up, best of two)); manipulation 345.7 (smoke-2 on the slice with the registered configuration: CPU pools at the quota, the ROIAlign kernel (Session E's follow-up, best of two)); body 94.3 (the stock body clip on the slice with the registered configuration: CPU pools at the quota, the ROIAlign kernel (Session E's follow-up, best of two)).

**Finding:** the Detectron2 detectors are much slower on the slice than on the Mac's CPU: Mask R-CNN took 1623 s for the body clip against 32 s, and Hands23 221 s against 10 s, while SAM 2.1 ran faster (84 s against 460 s for the body track). Diagnosed in Session E (below): SCRFD's time was onnxruntime's thread pool oversubscribing the CPU quota, and the Detectron2 detectors' was torchvision's deterministic Python roi_align. In the costs above, Mask R-CNN is 1623 s of the body path's 1708 s and Hands23 204 s of the manipulation reference's 551 s; SCRFD on onnxruntime's CPU provider is 82 s of the face reference's 99 s, and its thread pool is sized to the host, not the pod's CPU quota.

GPU time for generation and masks only; replacements, metrics, debugging and pod setup are extra (protocol section 12, equation 30).

## Diagnostics

### After Session B

- smoke-1, iteration 0 in one process: 613 module calls each; first differing call: none. Dense and HSA outputs by attention backend: default equal; flash only equal; efficient only equal; cudnn only equal.
- smoke-2, iteration 0 in one process: 613 module calls each; first differing call: none. Dense and HSA outputs by attention backend: default equal; flash only equal; efficient only equal; cudnn only equal.
- Attention on identical tensors, every query vs a subset: flash row-invariant (max diff 0); efficient row-invariant (max diff 0); cudnn not row-invariant (max diff 0.000244).
- Flash pinned, smoke-2, full trajectories: HSA all-full vs dense bitwise yes; C3 fast path vs oracle bitwise yes.
- Across processes (default backend in a new process vs generate's run 1, attention CUDNN_ATTENTION): bitwise no, max abs diff 3.76, first divergent step 0; 58 s.
- Across processes (flash in a new process vs the saved flash run, attention FLASH_ATTENTION): bitwise yes, max abs diff 0, first divergent step none; 78 s.

### Mask detectors and CPU threads on the slice (Session E)

Host d3e08104aa02: cgroup cpu.max `340000 100000` of 128 host CPUs. Each configuration ran in its own process. session_c is Session C's setup (env.sh's OMP and MKL threads, the quota rounded up; onnxruntime, OpenCV and torch's inter-op pool at their defaults); capped_N sets every pool to N threads (torch's inter-op and onnxruntime's inter-op to 1).

| Config | Pools (intra/inter) | SCRFD (s/frame) | WiLoR (s/frame) | Hands23 (s/keyframe) | Mask R-CNN (s/frame) | SCRFD: throttled periods | Mask R-CNN CPU (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| session_c | torch 4/64, onnxruntime default, cv2 1; 78 OS threads | 0.784 | 0.00857 | 16 | 20.5 | 635 of 635 | 184 |
| capped_3 | torch 3/1, onnxruntime 3, cv2 1; 15 OS threads | 0.0667 | 0.00815 | 16 | 19.8 | 0 of 54 | 177 |
| capped_4 | torch 4/1, onnxruntime 4, cv2 1; 21 OS threads | 0.0634 | 0.00795 | 15.4 | 19.7 | 52 of 52 | 177 |
| single | torch 1/1, onnxruntime 1, cv2 1; 3 OS threads | 0.149 | 0.00803 | 15.4 | 19.7 | 0 of 121 | 177 |

Outputs grouped by bitwise equality: scrfd [session_c, capped_3, capped_4, single]; wilor [session_c, capped_3, capped_4, single]; hands23 [session_c, capped_3, capped_4, single]; mask_rcnn [session_c, capped_3, capped_4, single].

| Config | Detectron2 model | Preprocessing (s) | Model (s) | Copy to host (s) |
| --- | --- | --- | --- | --- |
| session_c | mask_rcnn | 0.00621 | 19.9 | 0.000222 |
| session_c | hands23 | 0.006 | 18.9 | 0.000268 |
| capped_3 | mask_rcnn | 0.00613 | 19.2 | 0.000245 |
| capped_3 | hands23 | 0.00613 | 18.2 | 0.000242 |
| capped_4 | mask_rcnn | 0.00645 | 19.2 | 0.000233 |
| capped_4 | hands23 | 0.00858 | 18.2 | 0.000187 |
| single | mask_rcnn | 0.0083 | 19.2 | 0.000201 |
| single | hands23 | 0.00756 | 18.2 | 0.000164 |

Deterministic algorithms off (a diagnostic only), capped_3: mask_rcnn 0.09 s a call, same outputs no; hands23 0.14 s a call, same outputs no.

Mask R-CNN, one call profiled at session_c (top operators by self CPU time):

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                  cudaStreamSynchronize        99.91%       21.743s        99.91%       21.743s     472.684ms       0.000us         0.00%       0.000us       0.000us            46  
                                       cudaLaunchKernel         0.02%       4.443ms         0.02%       4.443ms       3.552us       0.000us         0.00%       0.000us       0.000us          1251  
                                aten::cudnn_convolution         0.01%       1.341ms         0.01%       2.238ms      27.625us      43.057ms         0.20%      43.057ms     531.570us            81  
                                         aten::_to_copy         0.01%       1.312ms         0.19%      41.378ms       1.591ms       0.000us         0.00%     333.469us      12.826us            26  
                                            aten::empty         0.00%     756.735us         0.01%       1.695ms       4.414us       0.000us         0.00%       1.936ms       5.042us           384  
## Call CompiledFxGraph fdyk6cadriulhscyd3lh52kwoncd...         0.00%     748.667us         0.01%       1.191ms     238.206us       0.000us         0.00%       21.681s        4.336s             5  
                                        cudaMemcpyAsync         0.00%     684.841us         0.00%     684.841us      12.682us       0.000us         0.00%       0.000us       0.000us            54  
                                            aten::fill_         0.00%     679.203us         0.01%       2.067ms       4.297us       2.166ms         0.01%       2.166ms       4.504us           481  
                             Torch-Compiled Region: 0/0         0.00%     515.869us         0.01%       1.922ms     384.474us       0.000us         0.00%       21.681s        4.336s             5  
```

Mask R-CNN, one call profiled at capped_3 (top operators by self CPU time):

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                  cudaStreamSynchronize        99.91%       20.930s        99.91%       20.930s     454.990ms       0.000us         0.00%       0.000us       0.000us            46  
                                       cudaLaunchKernel         0.02%       4.073ms         0.02%       4.073ms       3.256us       0.000us         0.00%       0.000us       0.000us          1251  
                                aten::cudnn_convolution         0.01%       1.352ms         0.01%       2.273ms      28.066us      41.966ms         0.20%      41.966ms     518.104us            81  
                                        cudaMemcpyAsync         0.00%     762.754us         0.00%     762.754us      14.125us       0.000us         0.00%       0.000us       0.000us            54  
## Call CompiledFxGraph ff2f2vpfiybzjchonvvgtykxm2tp...         0.00%     761.872us         0.01%       1.265ms     252.912us       0.000us         0.00%       20.867s        4.173s             5  
                                            aten::empty         0.00%     743.191us         0.01%       1.721ms       4.481us       0.000us         0.00%       1.917ms       4.991us           384  
                                            aten::fill_         0.00%     698.062us         0.01%       1.521ms       3.163us       2.139ms         0.01%       2.139ms       4.447us           481  
                             Torch-Compiled Region: 0/0         0.00%     542.077us         0.01%       2.029ms     405.814us       0.000us         0.00%       20.867s        4.173s             5  
                                          aten::nonzero         0.00%     475.422us        99.41%       20.825s        1.041s     127.038us         0.00%     154.589us       7.729us            20  
```

| Run | Reference | Stratum | Seconds (models built) | Components (s) | Masks equal to session_c |
| --- | --- | --- | --- | --- | --- |
| session_c | smoke-1 | face | 115 | detect_track 113, roi_coverage_measurement_eligibility 1, F.detector 98, F.sam 10 |  |
| session_c | smoke-2 | manipulation | 528 | detect_track 527, roi_coverage_measurement_eligibility 1, H.detector 2, H.sam 282, O.hands23 184, O.sam 54 |  |
| capped_3 | smoke-1 | face | 20.8 | detect_track 20, roi_coverage_measurement_eligibility 1, F.detector 6, F.sam 9 | yes |
| capped_3 | smoke-2 | manipulation | 532 | detect_track 531, roi_coverage_measurement_eligibility 1, H.detector 2, H.sam 282, O.hands23 189, O.sam 54 | yes |
| capped_3_body |  |  | Mask R-CNN takes 19.782549780896968 s a frame at capped_3 in the bench; the body clip would take about 27 min |  |  |
| capped_4_body |  |  | Mask R-CNN takes 19.689176184642648 s a frame at capped_4 in the bench; the body clip would take about 27 min |  |  |

The Mac's CPU in its validated configuration (torch 8 threads, its default; onnxruntime's default), the same references:

| Reference | Stratum | Seconds | Components (s) |
| --- | --- | --- | --- |
| smoke-1 | face | 58.5 | detect_track 58, roi_coverage_measurement_eligibility 1, F.detector 5, F.sam 51 |
| smoke-2 | manipulation | 2.08e+03 | detect_track 2076, roi_coverage_measurement_eligibility 1, H.detector 5, H.sam 1752, O.hands23 10, O.sam 305 |
| body | body | 502 | detect_track 501, roi_coverage_measurement_eligibility 1, B.detector 33, B.sam 466 |

### Follow-up: Detectron2's ROIAlign through torchvision's kernel (Session E)

With deterministic algorithms on and the input on CUDA, torchvision 0.29.1 replaces its roi_align kernel with a pure-Python version, lazily torch.compile'd, which exists for a deterministic backward pass; the profile above shows it taking the whole Mask R-CNN call. The masks need only the forward pass. This run called the kernel for inference, with every other operation deterministic and the CPU pools at 3.

- mask_rcnn: 0.080 s a call with the kernel; repeat bitwise yes; bitwise equal to the Python version on the first 3 inputs: no, no, no. Against it: {"same_count": true, "max_box_diff_px": 3.0517578125e-05, "max_score_diff": 0.0, "min_mask_iou": 1.0}; {"same_count": true, "max_box_diff_px": 0.0, "max_score_diff": 2.384185791015625e-07, "min_mask_iou": 1.0}; {"same_count": true, "max_box_diff_px": 9.1552734375e-05, "max_score_diff": 0.0, "min_mask_iou": 0.9999667066187242}.
- hands23: 0.147 s a call with the kernel; repeat bitwise yes; bitwise equal to the Python version on the first 3 inputs: no, no, no. Against it: {"same_count": true, "same_labels": true, "max_box_diff_px": 2.288818359375e-05, "max_hand_score_diff": 0.0}; {"same_count": true, "same_labels": true, "max_box_diff_px": 0.0001220703125, "max_hand_score_diff": 0.0}; {"same_count": true, "same_labels": true, "max_box_diff_px": 6.103515625e-05, "max_hand_score_diff": 0.0}.

| Reference | Stratum | Seconds (best of 2) | Repeat identical | Components (s) |
| --- | --- | --- | --- | --- |
| smoke-1 | face | 18.0 | yes | detect_track 18, roi_coverage_measurement_eligibility 1, F.detector 6, F.sam 9 |
| smoke-2 | manipulation | 346 | yes | detect_track 345, roi_coverage_measurement_eligibility 1, H.detector 2, H.sam 283, O.hands23 2, O.sam 54 |
| body | body | 94.3 | yes | detect_track 93, roi_coverage_measurement_eligibility 1, B.detector 6, B.sam 84 |

#### Seconds per reference: the slice against the Mac

| Stratum | Reference | Slice, threads capped (s) | Slice / Mac | Slice, also the ROIAlign kernel (s) | Slice / Mac | Mac (s) |
| --- | --- | --- | --- | --- | --- | --- |
| face | smoke-1 | 20.8 | 0.356 | 18.0 | 0.307 | 58.5 |
| manipulation | smoke-2 | 532 | 0.256 | 346 | 0.166 | 2077 |
| body | body | 94.3 (Session C) | 0.188 | 94.3 | 0.188 | 502 |

Weighted by the main run's references (72 face, 72 manipulation, 16 body, 16 control): 11.5 h on the slice with the threads capped, 7.7 h with the ROIAlign kernel too, against 44.9 h on the Mac. Rule: the slice should come within about 2x of the Mac's per-reference mask time (Chase, 2026-10-07).

- du at the start: allocated 57.7 GB in 7 s; apparent 48.3 GB in 7 s (stop at 90 GB).
- du at the end: allocated 57.7 GB in 10 s; apparent 48.3 GB in 7 s (stop at 90 GB).

## Warnings

Operations without a deterministic implementation (warn-only mode): 0.
Other warnings: 3 distinct (listed in the JSON).

Saved inputs, references, sheets, masks and logs are not versioned; their SHA-256 hashes are in `smoke_test.sha256`.
