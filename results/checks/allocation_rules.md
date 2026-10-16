# Allocation rules

**Overall: PASS** (81/81 judged checks passed)

Code `ac677363bf3a475554470b9b1312a35e5d60a585` (dirty: False); torch 2.14.1, diffusers 0.40.0, macOS-26.3.1-arm64-arm-64bit.

| Test | Model | Device | Dtype | Case | Result |
| --- | --- | --- | --- | --- | --- |
| T8 need score hand example | unit | cpu |  |  | PASS |
| T8 stops on NaN and infinity | unit | cpu |  |  | PASS |
| T8 rejects K < 2 and non-fp32 input | unit | cpu |  |  | PASS |
| T8 K is the all-active prefix for every preset | unit |  |  |  | PASS |
| T8 need score hand example | unit | mps |  |  | PASS |
| T8 stops on NaN and infinity | unit | mps |  |  | PASS |
| T8 rejects K < 2 and non-fp32 input | unit | mps |  |  | PASS |
| T8 K is the all-active prefix for every preset | unit |  |  |  | PASS |
| T9 C3 ties | unit |  |  |  | PASS |
| T9 C4 ties | unit |  |  |  | PASS |
| T9 C5 ties | unit |  |  |  | PASS |
| T9 C6 ties | unit |  |  |  | PASS |
| T9 group sizes filled exactly | unit |  |  |  | PASS |
| T9 registered group sizes at N = 32,760 | unit |  |  | HSA-50 | PASS |
| T9 registered group sizes at N = 32,760 | unit |  |  | HSA-75B | PASS |
| T9 registered group sizes at N = 32,760 | unit |  |  | switch-20 | PASS |
| T9 registered group sizes at N = 32,760 | unit |  |  | switch-30 | PASS |
| T12 fractional mask | unit |  |  | lambda 1 | PASS |
| T12 fractional mask | unit |  |  | lambda 3 | PASS |
| T12 fractional mask | unit |  |  | lambda 10 | PASS |
| T12 uniform mask: masked tokens only rise | unit |  |  | lambda 1 | PASS |
| T12 uniform mask: masked tokens only rise | unit |  |  | lambda 3 | PASS |
| T12 uniform mask: masked tokens only rise | unit |  |  | lambda 10 | PASS |
| T12 literal claim fails for fractional coverage (counterexample) | unit |  |  |  | info |
| T13 | unit |  |  |  | PASS |
| T14 C6 binary first, then g | unit |  |  |  | PASS |
| T15 half-covered cell gives 0.5 | unit |  |  |  | PASS |
| T15 frame groups map exactly | unit |  |  |  | PASS |
| T15 dilation is spatial only, zero outside the grid | unit |  |  |  | PASS |
| T15 shapes at tiny, small and full size | unit |  |  |  | PASS |
| T15 token order matches the Transformer | unit |  |  |  | PASS |
| T16 hand example (P, R, R_norm, lift, S_ROI, A_ROI) | unit |  |  |  | PASS |
| T16 no-op only when every label matches | unit |  |  |  | PASS |
| T16 empty ROI reports N/A | unit |  |  |  | PASS |
| T17 labels assigned once, then frozen | unit |  |  |  | PASS |
| T18 ascending sort fails T9 | unit |  |  |  | PASS |
| T18 g-seeded C5 permutation fails T13 | unit |  |  |  | PASS |
| T8 run uses guided v at n = 0..K-1 | tiny | cpu | fp32 |  | PASS |
| T18 conditional-only g fails T8 | tiny | cpu | fp32 |  | PASS |
| registry: C1 equals the dense loop | tiny | cpu | fp32 |  | PASS |
| T10 lambda = 0 equals C3 | tiny | cpu | fp32 |  | PASS |
| T11 empty mask equals C3 | tiny | cpu | fp32 | lambda 1 | PASS |
| T11 empty mask equals C3 | tiny | cpu | fp32 | lambda 3 | PASS |
| T11 empty mask equals C3 | tiny | cpu | fp32 | lambda 10 | PASS |
| T17 equal active counts and budgets across C3-C6 and A1 | tiny | cpu | fp32 |  | PASS |
| registry: C2 runs 20 dense steps at fraction 0.5 | tiny | cpu | fp32 |  | PASS |
| T8 run uses guided v at n = 0..K-1 | tiny | cpu | bf16 |  | PASS |
| T18 conditional-only g fails T8 | tiny | cpu | bf16 |  | PASS |
| registry: C1 equals the dense loop | tiny | cpu | bf16 |  | PASS |
| T10 lambda = 0 equals C3 | tiny | cpu | bf16 |  | PASS |
| T11 empty mask equals C3 | tiny | cpu | bf16 | lambda 1 | PASS |
| T11 empty mask equals C3 | tiny | cpu | bf16 | lambda 3 | PASS |
| T11 empty mask equals C3 | tiny | cpu | bf16 | lambda 10 | PASS |
| T17 equal active counts and budgets across C3-C6 and A1 | tiny | cpu | bf16 |  | PASS |
| registry: C2 runs 20 dense steps at fraction 0.5 | tiny | cpu | bf16 |  | PASS |
| T8 run uses guided v at n = 0..K-1 | tiny | mps | fp32 |  | PASS |
| T18 conditional-only g fails T8 | tiny | mps | fp32 |  | PASS |
| registry: C1 equals the dense loop | tiny | mps | fp32 |  | PASS |
| T10 lambda = 0 equals C3 | tiny | mps | fp32 |  | PASS |
| T11 empty mask equals C3 | tiny | mps | fp32 | lambda 1 | PASS |
| T11 empty mask equals C3 | tiny | mps | fp32 | lambda 3 | PASS |
| T11 empty mask equals C3 | tiny | mps | fp32 | lambda 10 | PASS |
| T17 equal active counts and budgets across C3-C6 and A1 | tiny | mps | fp32 |  | PASS |
| registry: C2 runs 20 dense steps at fraction 0.5 | tiny | mps | fp32 |  | PASS |
| T8 run uses guided v at n = 0..K-1 | tiny | mps | bf16 |  | PASS |
| T18 conditional-only g fails T8 | tiny | mps | bf16 |  | PASS |
| registry: C1 equals the dense loop | tiny | mps | bf16 |  | PASS |
| T10 lambda = 0 equals C3 | tiny | mps | bf16 |  | PASS |
| T11 empty mask equals C3 | tiny | mps | bf16 | lambda 1 | PASS |
| T11 empty mask equals C3 | tiny | mps | bf16 | lambda 3 | PASS |
| T11 empty mask equals C3 | tiny | mps | bf16 | lambda 10 | PASS |
| T17 equal active counts and budgets across C3-C6 and A1 | tiny | mps | bf16 |  | PASS |
| registry: C2 runs 20 dense steps at fraction 0.5 | tiny | mps | bf16 |  | PASS |
| registry: C1 reproduces the committed R1 frames | real | mps | bf16 |  | PASS |
| T8 run uses guided v at n = 0..K-1 | real | mps | bf16 |  | PASS |
| T18 conditional-only g fails T8 | real | mps | bf16 |  | PASS |
| T10 lambda = 0 equals C3 | real | mps | bf16 |  | PASS |
| T11 empty mask equals C3 | real | mps | bf16 | lambda 1 | PASS |
| T11 empty mask equals C3 | real | mps | bf16 | lambda 3 | PASS |
| T11 empty mask equals C3 | real | mps | bf16 | lambda 10 | PASS |
| T17 equal active counts and budgets across C3-C6 and A1 | real | mps | bf16 |  | PASS |
| registry: C2 runs 20 dense steps at fraction 0.5 | real | mps | bf16 |  | PASS |

## Registered values

- Need-score floor 1e-08; ROI threshold 0.1; lambda grid 1, 3, 10 (0 only as a control).
- K (leading iterations in which every token runs): HSA-50 5, HSA-75B 10, switch-20 9, switch-30 12.
- Tie rules: C3: g descending, then token ID ascending; C4: (1 + lambda m) g descending (float64, exact), then token ID ascending; C5: m descending, then position in torch.randperm(N) seeded by the prompt ID; C6: b descending, then g descending, then token ID ascending.
- Small-resolution group sizes: registered high-group share, rounded (31 of 192 tiny tokens, 312 of 1,950 real tokens).
- Devices: The need score and every model test run on CPU and MPS. Ranking (numpy lexsort in float64), token mapping and the diagnostics run on CPU by design, so their results do not depend on the device.

## T12 note

The literal T12 statement, that masked tokens can only rise in rank relative to C3, is false when coverage is fractional: a masked token can be overtaken by a more-covered masked token. The judged invariants are the ones that hold for any coverage: equal group sizes, unmasked tokens keep their relative order and can only fall, no masked token falls behind an unmasked token it led, and the ROI's full-step count can only rise. The literal statement is judged for uniform coverage, where it holds.

## Sphere demo (informational, no quality claims)

Circle at x 213, y 126, radius 74 px on every frame: 615 of 1950 tokens in the binary ROI. Placeholder embeddings, synthetic mask, 416 x 240. Inside the mask means latent elements of tokens with b = 1[m >= 0.1], via unpatchify. Latent MSE is against R1 (C1).

| Condition | P | R | R_norm | Lift | A_ROI | S_ROI | Changed tokens | Mean m, full-step | MSE inside | MSE outside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | 0.04808 | 0.02439 | 0.04808 | 0.1524 | 0 | 0.2636 | 0 | 0.03384 | 0.003488 | 0.0008145 |
| C4 lambda 1 | 1 | 0.5073 | 1 | 3.171 | 0.4992 | 0.4479 | 604 | 0.9779 | 0.002611 | 0.0007976 |
| C4 lambda 3 | 1 | 0.5073 | 1 | 3.171 | 0.5057 | 0.4479 | 608 | 0.9929 | 0.0025 | 0.0007866 |
| C4 lambda 10 | 1 | 0.5073 | 1 | 3.171 | 0.5057 | 0.4479 | 608 | 0.9949 | 0.002318 | 0.0007568 |
| C5 | 1 | 0.5073 | 1 | 3.171 | 0.5154 | 0.4479 | 614 | 1 | 0.002568 | 0.0007722 |
| C6 | 1 | 0.5073 | 1 | 3.171 | 0.4829 | 0.4479 | 594 | 0.8123 | 0.003944 | 0.0006334 |

C4's ROI share of full-step slots rising with lambda is not testable at this mask size: the binary ROI (615 tokens) is larger than the full-step group (312 slots), so every full-step slot goes to the ROI from lambda = 1 on and P stays 1. The mean coverage of the full-step tokens is reported alongside, without interpretation.

## Real-model inputs

- Inputs match wan_real_mps.json: True; weights match: True.
- Demo images are not versioned; their hashes are in the JSON and `allocation_rules.sha256`.
