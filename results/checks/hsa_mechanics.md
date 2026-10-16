# HSA mechanics on the Wan2.1 Transformer

**Overall: PASS** (84/84 judged checks passed; acceptance gate, real-model T3: fast path vs oracle, bitwise, every label seed: PASS)

Code `9694b2722ec63d29991c74cfabb5b79875229968` (dirty: False); torch 2.14.1, diffusers 0.40.0, macOS-26.3.1-arm64-arm-64bit.
Shift 8, guidance 6, 40 steps. HSA-50 window and stride with the registered high-group share, rounded (31 of 192 tokens for the tiny model, 312 of 1,950 for the real model at 416 x 240).

| Test | Model | Device | Dtype | Case | Bitwise | Max abs diff | First divergence | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| schedule (N = 32,760) |  |  |  | HSA-50 |  |  |  | PASS |
| schedule (N = 32,760) |  |  |  | HSA-75B |  |  |  | PASS |
| schedule (N = 32,760) |  |  |  | switch-20 |  |  |  | PASS |
| schedule (N = 32,760) |  |  |  | switch-30 |  |  |  | PASS |
| T0 token locality | tiny | cpu | fp32 |  |  |  |  | PASS |
| T1 all groups full vs dense | tiny | cpu | fp32 |  | yes | 0 |  | PASS |
| T2 all active, layer by layer | tiny | cpu | fp32 |  | yes | 0 |  | PASS |
| oracle all groups full vs dense | tiny | cpu | fp32 |  | yes | 0 |  | PASS |
| oracle HSA-50-style differs from dense | tiny | cpu | fp32 |  | no | 0.0671 | 5 | PASS |
| T3 fast path vs oracle | tiny | cpu | fp32 | seed 0 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | cpu | fp32 | seed 0 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | cpu | fp32 | seed 1 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | cpu | fp32 | seed 1 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | cpu | fp32 | seed 2 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | cpu | fp32 | seed 2 |  |  |  | PASS |
| T5 shared KV cache | tiny | cpu | fp32 | seed 0 | no | 0.00182 | 5 | PASS |
| T6 no RoPE gather | tiny | cpu | fp32 | seed 0 | no | 0.00515 | 5 | PASS |
| T6 no cache refresh | tiny | cpu | fp32 | seed 0 | no | 0.0128 | 1 | PASS |
| T0 token locality | tiny | cpu | bf16 |  |  |  |  | PASS |
| T1 all groups full vs dense | tiny | cpu | bf16 |  | yes | 0 |  | PASS |
| T2 all active, layer by layer | tiny | cpu | bf16 |  | yes | 0 |  | PASS |
| oracle all groups full vs dense | tiny | cpu | bf16 |  | yes | 0 |  | PASS |
| oracle HSA-50-style differs from dense | tiny | cpu | bf16 |  | no | 0.0736 | 5 | PASS |
| T3 fast path vs oracle | tiny | cpu | bf16 | seed 0 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | cpu | bf16 | seed 0 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | cpu | bf16 | seed 1 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | cpu | bf16 | seed 1 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | cpu | bf16 | seed 2 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | cpu | bf16 | seed 2 |  |  |  | PASS |
| T5 shared KV cache | tiny | cpu | bf16 | seed 0 | no | 0.0464 | 5 | PASS |
| T6 no RoPE gather | tiny | cpu | bf16 | seed 0 | no | 0.0301 | 5 | PASS |
| T6 no cache refresh | tiny | cpu | bf16 | seed 0 | no | 0.0384 | 1 | PASS |
| T4 synthetic, full size | synthetic | cpu |  | HSA-50 | yes | 0 |  | PASS |
| T4 synthetic, full size | synthetic | cpu |  | HSA-75B | yes | 0 |  | PASS |
| T4 synthetic, full size | synthetic | cpu |  | switch-20 | yes | 0 |  | PASS |
| T4 synthetic, full size | synthetic | cpu |  | switch-30 | yes | 0 |  | PASS |
| T4 control: varying low-group velocity differs | synthetic | cpu |  | HSA-50 | no | 0.0437 | 5 | PASS |
| T0 token locality | tiny | mps | fp32 |  |  |  |  | PASS |
| T1 all groups full vs dense | tiny | mps | fp32 |  | yes | 0 |  | PASS |
| T2 all active, layer by layer | tiny | mps | fp32 |  | yes | 0 |  | PASS |
| oracle all groups full vs dense | tiny | mps | fp32 |  | yes | 0 |  | PASS |
| oracle HSA-50-style differs from dense | tiny | mps | fp32 |  | no | 0.0671 | 5 | PASS |
| T3 fast path vs oracle | tiny | mps | fp32 | seed 0 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | mps | fp32 | seed 0 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | mps | fp32 | seed 1 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | mps | fp32 | seed 1 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | mps | fp32 | seed 2 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | mps | fp32 | seed 2 |  |  |  | PASS |
| T5 shared KV cache | tiny | mps | fp32 | seed 0 | no | 0.00182 | 5 | PASS |
| T6 no RoPE gather | tiny | mps | fp32 | seed 0 | no | 0.00515 | 5 | PASS |
| T6 no cache refresh | tiny | mps | fp32 | seed 0 | no | 0.0128 | 1 | PASS |
| T0 token locality | tiny | mps | bf16 |  |  |  |  | PASS |
| T1 all groups full vs dense | tiny | mps | bf16 |  | yes | 0 |  | PASS |
| T2 all active, layer by layer | tiny | mps | bf16 |  | yes | 0 |  | PASS |
| oracle all groups full vs dense | tiny | mps | bf16 |  | yes | 0 |  | PASS |
| oracle HSA-50-style differs from dense | tiny | mps | bf16 |  | no | 0.0757 | 5 | PASS |
| T3 fast path vs oracle | tiny | mps | bf16 | seed 0 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | mps | bf16 | seed 0 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | mps | bf16 | seed 1 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | mps | bf16 | seed 1 |  |  |  | PASS |
| T3 fast path vs oracle | tiny | mps | bf16 | seed 2 | yes | 0 |  | PASS |
| T7 active counts and budget | tiny | mps | bf16 | seed 2 |  |  |  | PASS |
| T5 shared KV cache | tiny | mps | bf16 | seed 0 | no | 0.0421 | 5 | PASS |
| T6 no RoPE gather | tiny | mps | bf16 | seed 0 | no | 0.0313 | 5 | PASS |
| T6 no cache refresh | tiny | mps | bf16 | seed 0 | no | 0.0336 | 1 | PASS |
| T4 synthetic, full size | synthetic | mps |  | HSA-50 | yes | 0 |  | PASS |
| T4 synthetic, full size | synthetic | mps |  | HSA-75B | yes | 0 |  | PASS |
| T4 synthetic, full size | synthetic | mps |  | switch-20 | yes | 0 |  | PASS |
| T4 synthetic, full size | synthetic | mps |  | switch-30 | yes | 0 |  | PASS |
| T4 control: varying low-group velocity differs | synthetic | mps |  | HSA-50 | no | 0.0437 | 5 | PASS |
| T0 token locality | real | mps | bf16 |  |  |  |  | PASS |
| T1 all groups full vs dense | real | mps | bf16 |  | yes | 0 |  | PASS |
| T2 all active, layer by layer | real | mps | bf16 |  | yes | 0 |  | PASS |
| oracle all groups full vs dense | real | mps | bf16 |  | yes | 0 |  | PASS |
| oracle HSA-50-style differs from dense | real | mps | bf16 |  | no | 0.319 | 5 | PASS |
| T3 fast path vs oracle | real | mps | bf16 | seed 0 | yes | 0 |  | PASS |
| T7 active counts and budget | real | mps | bf16 | seed 0 |  |  |  | PASS |
| T3 fast path vs oracle | real | mps | bf16 | seed 1 | yes | 0 |  | PASS |
| T7 active counts and budget | real | mps | bf16 | seed 1 |  |  |  | PASS |
| T3 fast path vs oracle | real | mps | bf16 | seed 2 | yes | 0 |  | PASS |
| T7 active counts and budget | real | mps | bf16 | seed 2 |  |  |  | PASS |
| T5 shared KV cache | real | mps | bf16 | seed 0 | no | 0.2 | 5 | PASS |
| T6 no RoPE gather | real | mps | bf16 | seed 0 | no | 0.466 | 5 | PASS |
| T6 no cache refresh | real | mps | bf16 | seed 0 | no | 1.04 | 1 | PASS |

## Registered schedules (N = 32,760)

| Preset | High / low | Window | Stride | Evaluations (high, low) | Token-step fraction | Low-group iterations |
| --- | --- | --- | --- | --- | --- | --- |
| HSA-50 | 5,242 / 27,518 | [4, 36) | 4 | 40, 16 | 0.496007 | 0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 39 |
| HSA-75B | 10,811 / 21,949 | [10, 30) | 4 | 40, 25 | 0.748752 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 |
| switch-20 | 1,560 / 31,200 | [8, 36) | 4 | 40, 19 | 0.500000 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 39 |
| switch-30 | 1,560 / 31,200 | [12, 36) | 8 | 40, 19 | 0.500000 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 24, 32, 36, 37, 38, 39 |

## Why the oracle is exact: token-mixing operations

Source: `diffusers/models/transformers/transformer_wan.py (diffusers 0.40.0)`.

| Operation | Location | Mixes video tokens? |
| --- | --- | --- |
| patch_embedding | WanTransformer3DModel.__init__ (line 598) | no: Conv3d with kernel_size == stride == patch_size, no padding: each token sees only its own patch |
| rope | WanRotaryPosEmbed.forward (lines 395-416) | no: depends on the latent shape only; one frequency row per token |
| condition_embedder | WanTimeTextImageEmbedding.forward (lines 330-351) | no: timestep and text only; shared by every token |
| norm1, norm2, norm3, norm_out | FP32LayerNorm (normalization.py lines 85-93) | no: F.layer_norm over the channel dimension of each token |
| attn1 q/k/v, norm_q, norm_k, to_out | WanAttnProcessor.__call__ (lines 93-100, 160-161) | no: Linear and RMSNorm over the channel dimension of each token |
| attn1 attention | WanAttnProcessor.__call__ (lines 143-153) | **yes**: softmax(QK^T)V over all video tokens: the only token-mixing operation |
| attn2 (cross-attention) | WanTransformerBlock.forward (line 494) | no: queries are video tokens, keys and values are text tokens: no video token reads another |
| ffn | FeedForward.forward (attention.py), GELU (activations.py lines 87-90) | no: Linear, tanh-GELU, Linear on each token |
| modulation and residuals | WanTransformerBlock.forward (lines 483-502) | no: elementwise with a per-sample timestep table |
| proj_out and unpatchify | WanTransformer3DModel.forward (lines 723-730) | no: Linear on each token, then a reshape and permute that places each token's patch |

T0 checks this empirically: with self-attention made token-local, perturbing one token changes only that token's output.

## Fixture change (disclosed)

The HSA tests first used diffusers' unit-test tiny Transformer: 2 heads of 12 dims and 8 text tokens. Its judged T3 failed once, on MPS in bf16 for label seed 0 at step 30. At that head size the attention kernel's accumulation order depends on the number of queries, and the fast path and the oracle send different query counts. With identical q, K and V the kernel's outputs differ by one rounding step; the diagnostic below recomputes this evidence on every run. The fixture now uses the real model's attention geometry, 2 heads of 128 dims and 512 text tokens, so the tiny tests exercise the same kernels as the real model. Every tiny test, including the negative controls, is judged bitwise in fp32 and bf16. The real-model T3 remains the final acceptance gate.

## Head-dim-12 diagnostic (not judged)

- cpu, fp32: seed 0: max abs diff 2.38e-06 from step 5; seed 1: max abs diff 2.03e-06 from step 5; seed 2: max abs diff 1.67e-06 from step 5.
  - Seed 0, step 5: first differing module `blocks.0.attn2` (cond branch); module inputs equal on active rows: True; q, K, V equal: True. Attention with 192 queries, then selecting the active rows, vs with 31 queries: 12 of 744 elements differ, max 1.49e-08; same inputs in fp32: max 1.49e-08.
- cpu, bf16: seed 0: bitwise; seed 1: bitwise; seed 2: bitwise.
- mps, fp32: seed 0: max abs diff 1.91e-06 from step 5; seed 1: max abs diff 1.91e-06 from step 5; seed 2: max abs diff 1.91e-06 from step 5.
  - Seed 0, step 5: first differing module `blocks.0.attn1` (cond branch); module inputs equal on active rows: True; q, K, V equal: True. Attention with 192 queries, then selecting the active rows, vs with 31 queries: 226 of 744 elements differ, max 1.19e-07; same inputs in fp32: max 1.19e-07.
- mps, bf16: seed 0: max abs diff 0.0238 from step 30; seed 1: bitwise; seed 2: bitwise.
  - Seed 0, step 30: first differing module `blocks.1.attn1` (cond branch); module inputs equal on active rows: True; q, K, V equal: True. Attention with 192 queries, then selecting the active rows, vs with 31 queries: 1 of 744 elements differ, max 0.00195; same inputs in fp32: max 1.19e-07.

## Real-model inputs

- `inputs.pt` sha256 `7bef2cd062b2197e35a33afe6b91503341b6d44212af39e59d8ff4fb89694b98`; matches wan_real_mps.json: True; tensor hashes match: True.
- Weights match wan_real_mps.json: True.
- Seconds: dense 104, skipping_tests 653
