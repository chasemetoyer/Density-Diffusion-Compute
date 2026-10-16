# Real Wan2.1-T2V-1.3B through the custom Euler loop (MPS)

**Overall: PASS**

Wan-AI/Wan2.1-T2V-1.3B-Diffusers @ `0fad780a534b6463e45facd96134c9f345acfa5b`; torch 2.14.1, diffusers 0.40.0, macOS-26.3.1-arm64-arm-64bit
Settings: shift 8, guidance 6, 40 steps, 416x240, 17 frames, placeholder text embeddings.

| Check | Result | Details |
| --- | --- | --- |
| finite | PASS |  |
| repeat_run | measured | bitwise_equal True, max_abs_diff 0, decoded_psnr_db inf |
| pipeline_fp32_velocity_latents | PASS | bitwise_equal True, max_abs_diff 0, tolerance 0 |
| decode_helper_vs_pipeline | PASS | max_abs_diff 0, tolerance 0, decoded_psnr_db inf |

## Information only

- Stock bf16 pipeline vs custom loop: final latent max abs diff 0.4181 (relative 0.2837), decoded PSNR 33.88 dB (protocol repeat-run threshold 45 dB).
- Neighbor differences (lower is smoother): r1: spatial 0.004881, temporal 0.01997; r3_stock_bf16: spatial 0.00454, temporal 0.02032; decoded_noise_latent: spatial 0.03023, temporal 0.12
- Velocity at the first two timesteps: velocity_rms 1.101, text_effect_rms 0.07847, timestep_effect_rms 0.01988, informative True
- Seconds per run: r1_custom 154.3, r2_custom 141.2, r3_stock_bf16 114.8, r4_pipeline_fp32_velocity 113.7
- Transformer parameter dtypes: {'float32': 3127296, 'bfloat16': 1415869504}

Contact sheet: `results/checks/wan_real_mps/contact_sheet.png`. MPS repeatability does not carry over to CUDA.

Saved inputs, frames, and previews are not versioned; their SHA-256 hashes are in the JSON and in `wan_real_mps.sha256` (check with `shasum -a 256 -c wan_real_mps.sha256` from `results/checks`).
