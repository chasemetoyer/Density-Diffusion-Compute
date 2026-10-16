# Milestone 5 check: metrics, prompt set, runner and the frozen analysis

Run on the Mac at commit `601fc6cbfd6f704014ca1b384cd7658583dbb0db` (dirty: False), 2026-10-07T17:46:07Z. Timings are the Mac's. Every dry-run number describes stand-ins (stock clips with a seeded disturbance, tiny random models), not Wan; dry-run reviews were automatic and are not Chase's.

## Part 1: prompt tooling

| List | Prompts | Registered | Commit |
| --- | --- | --- | --- |
| vbench_human | 100 | 100 | `fd18b3d` |
| vbench_human_action | 100 | 100 | `fd18b3d` |
| vbench_scenery | 100 | 100 | `fd18b3d` |
| vbench_architecture | 100 | 100 | `fd18b3d` |
| vbench2_human_anatomy | 120 | 120 | `fd18b3d` |
| vbench2_human_identity | 46 | 46 | `fd18b3d` |
| vbench2_human_clothes | 75 | 75 | `fd18b3d` |
| t2v_compbench_object_interactions | 200 | 200 | `dd5eff7` |
| t2v_compbench_action_binding | 200 | 200 | `dd5eff7` |

Validator on the mock prompt files: pilot 0 errors, main 0 errors (longest 21 T5 tokens).

`configs/prompts.csv`: 151 rows, drafted by the registered procedure; Chase has not reviewed it yet. The pilot's check (holdout rows): passes. The full check: primary manipulation: 31 prompts, registered 36; manipulation reserves: 0, at least 30 required.

## Part 2: metrics

Weight files verified: 12 of 12. Encode canary: passed (registered on Chases-MacBook-Pro.local (arm64, macOS-26.3.1-arm64-arm-64bit)).

### Encoding and storage per video (81 frames, 832 x 480)

| Clip | MP4 PSNR vs lossless (dB) | MP4 | FFV1 (lossless) | PNG frames | Same decoded frames on re-encode |
| --- | --- | --- | --- | --- | --- |
| face | 47.62 | 1.45 MB | 14.46 MB | 22.47 MB | True |
| hands | 47.03 | 2.29 MB | 17.06 MB | 24.30 MB | True |
| body | 44.93 | 2.55 MB | 17.22 MB | 25.58 MB | True |
| landscape | 44.34 | 7.64 MB | 26.80 MB | 46.63 MB | True |

### Compression in the guardrails' units (MP4 against its lossless frames)

| Clip | Full-frame LPIPS | Outside-ROI LPIPS | Masked LPIPS | Largest quality change (points) |
| --- | --- | --- | --- | --- |
| face | 0.0059 | 0.0063 | 0.0050 | -0.496 (aesthetic_quality) |
| hands | 0.0134 | 0.0156 | 0.0065 | -0.168 (imaging_quality) |
| body | 0.0053 | 0.0060 | 0.0025 | -0.174 (background_consistency) |
| landscape | 0.0075 | 0.0075 | n/a | -0.976 (background_consistency) |

Limits: 0.01 LPIPS and 0.5 points. In absolute terms the CRF 10 encoding is not negligible next to them, so every guardrail is measured before compression (registered by Chase on 2026-10-07): LPIPS inside each generation job and the quality dimensions on the lossless frames. The MP4's quality scores are recorded beside them as a descriptive check.

### Storage

Mean per video: PNG frames 29.7 MB, FFV1 18.9 MB, MP4 3.5 MB. With references in PNG and every condition in FFV1 (registered), the pilot adds about 2.4 GB and the main run 23.2 GB, so the volume would hold about 85.5 GB after the main run (du allocated; 125 GB volume, 112 GB stop; volume now 57.7 GB; metric weights 2.2 GB; stock-clip sizes stand in for Wan's frames; no replacements).

### Metric canary (on the Mac's CPU)

Recorded from two fresh processes: True (79 s); checked from a third: True (39 s). The pods record their own at their first session.

### Known answers and determinism

- Still video: subject_consistency 100.0000, background_consistency 100.0000, motion_smoothness 99.7596, aesthetic_quality 20.8159, imaging_quality 14.9616.
- Alternating frames: motion smoothness 73.506 against the expected 73.507.
- Two separate processes, identical scores (quality, LPIPS, pixel and detector checks): True (cpu (the Mac); CUDA is first checked on the pods).

## Parts 3, 4 and 7: the mock pilot, its rules and registration, and a small mock main run

- Mock pilot: {'embed': {'done': 12}, 'reference': {'done': 24}, 'mask': {'done': 24}, 'condition': {'done': 50}, 'score': {'done': 10}}. Setting sh8g6 (tie: True; every mock reference of a stratum is the same clip, so the rule only takes its tie path here; the other path is unit-tested).
- Pilot rules: lambda 3.0, C2 ratio 0.511, no-op at every lambda [], excluded holdout prompts ['6', '12']; stops: none (artifacts of the stand-in disturbance, not findings).
- The encoding's effect on the paired quality degradation, on stand-ins (mean over prompts, MP4 minus lossless): subject_consistency +0.025, background_consistency -0.047, motion_smoothness +0.002, aesthetic_quality -0.276, imaging_quality -0.045.
- Registration draft: 110 registered values; pilot slots complete.
- Mock main run: {'embed': {'done': 8}, 'reference': {'done': 14}, 'mask': {'done': 14}, 'condition': {'done': 58}, 'score': {'done': 2, 'ready': 8}}; grid blocks ['75 percent', 'A1', 'C5 and C6', 'controls', 'full body', 'primary seed 1', 'primary seed 2', 'switch points']; collected 20 primary rows and blocks {'C5 and C6': 16, 'A1': 6, '75 percent': 8, 'switch points': 16, 'full body': 2, 'controls': 2}.

### Planned cost: pilot

| Block | Jobs | RE | Machine | Hours | Dollars |
| --- | --- | --- | --- | --- | --- |
| embeddings | 12 | 0 | h100 | 0.00 | 0.00 |
| references (holdout) | 24 | 24 | h100 | 0.49 | 1.71 |
| masks (holdout) | 24 | 0 | slice | 1.21 | 0.72 |
| pilot | 60 | 30 | h100 | 0.65 | 2.27 |
| scores | 12 | 0 | slice | 0.00 | 0.00 |

Totals: h100 54 RE, 1.1 h, $3.98; slice 0 RE, 1.2 h, $0.72. Rates {'h100': 3.48, 'slice': 0.59} per hour; times from smoke test Sessions D (H100, flash attention) and E (slice, thread cap and ROIAlign kernel); C2, HSA-75B and the switches scaled by token-step fraction. Not included: metric scoring time (a placeholder until measured), embeddings, canaries, pod setup, replacements and debugging.

### Planned cost: main

| Block | Jobs | RE | Machine | Hours | Dollars |
| --- | --- | --- | --- | --- | --- |
| embeddings | 104 | 0 | h100 | 0.00 | 0.00 |
| references (primary) | 144 | 144 | h100 | 2.95 | 10.27 |
| masks (primary) | 144 | 0 | slice | 7.28 | 4.30 |
| references (secondary) | 16 | 16 | h100 | 0.33 | 1.14 |
| masks (secondary) | 16 | 0 | slice | 0.42 | 0.25 |
| references (control) | 16 | 16 | h100 | 0.33 | 1.14 |
| masks (control) | 16 | 0 | slice | 0.00 | 0.00 |
| primary seed 1 | 216 | 108 | h100 | 2.32 | 8.08 |
| C5 and C6 | 72 | 36 | h100 | 0.79 | 2.76 |
| 75 percent | 32 | 24 | h100 | 0.49 | 1.71 |
| switch points | 64 | 32 | h100 | 0.70 | 2.45 |
| scores | 176 | 0 | slice | 0.00 | 0.00 |
| A1 | 36 | 18 | h100 | 0.40 | 1.38 |
| primary seed 2 | 144 | 72 | h100 | 1.58 | 5.51 |
| full body | 32 | 16 | h100 | 0.35 | 1.22 |
| controls | 32 | 16 | h100 | 0.35 | 1.22 |

Totals: h100 498 RE, 10.6 h, $36.89; slice 0 RE, 7.7 h, $4.54. Rates {'h100': 3.48, 'slice': 0.59} per hour; times from smoke test Sessions D (H100, flash attention) and E (slice, thread cap and ROIAlign kernel); C2, HSA-75B and the switches scaled by token-step fraction. Not included: metric scoring time (a placeholder until measured), embeddings, canaries, pod setup, replacements and debugging.

## Part 5: the frozen analysis

- Mock 72-prompt main analysis through the CLI (freeze checked, records collected): GO (median at least 10 percent, lower bound above zero, and all seven guardrails pass); median 0.1462, 95% interval [0.13315674141639078, 0.15997644756692736].
- 15 percent improvement: GO (expected GO).
- No effect, 50 replicates: {'NO-GO': 50} (never GO: True).
- Injected harm: outside_lpips +0.02 gives NO-GO (confirmed harm), imaging_quality +1.0 gives NO-GO (confirmed harm).
- Missing score: BLOCKED.

