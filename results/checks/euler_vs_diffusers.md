# Euler loop vs diffusers FlowMatchEulerDiscreteScheduler

**Overall: PASS** (32/32 judged checks passed)

torch 2.14.1, diffusers 0.40.0, numpy 2.2.6, Python 3.10.19, macOS-26.3.1-arm64-arm-64bit

| Check | Device | Precision | Shift | Steps | Guidance | Bitwise | Max abs diff | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| schedule |  |  | 8 | 40 |  | yes | 0 |  | PASS |
| synthetic | cpu |  | 8 | 40 |  | yes | 0 | 0 | PASS |
| pipeline | cpu | float32 | 8 | 40 | 6 | yes | 0 | 0 | PASS |
| pipeline | cpu | bfloat16 | 8 | 40 | 6 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | cpu | bf16, stock latents bfloat16 | 8 | 40 | 6 | no | 0.156 |  | info |
| synthetic | mps |  | 8 | 40 |  | yes | 0 | 0 | PASS |
| pipeline | mps | float32 | 8 | 40 | 6 | yes | 0 | 0 | PASS |
| pipeline | mps | bfloat16 | 8 | 40 | 6 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | mps | bf16, stock latents bfloat16 | 8 | 40 | 6 | no | 0.156 |  | info |
| negative_control | cpu |  | 8 | 40 | 6 | no | 0.162 |  | PASS |
| schedule |  |  | 8 | 20 |  | yes | 0 |  | PASS |
| synthetic | cpu |  | 8 | 20 |  | yes | 0 | 0 | PASS |
| pipeline | cpu | float32 | 8 | 20 | 6 | yes | 0 | 0 | PASS |
| pipeline | cpu | bfloat16 | 8 | 20 | 6 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | cpu | bf16, stock latents bfloat16 | 8 | 20 | 6 | no | 0.0729 |  | info |
| synthetic | mps |  | 8 | 20 |  | yes | 0 | 0 | PASS |
| pipeline | mps | float32 | 8 | 20 | 6 | yes | 0 | 0 | PASS |
| pipeline | mps | bfloat16 | 8 | 20 | 6 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | mps | bf16, stock latents bfloat16 | 8 | 20 | 6 | no | 0.0794 |  | info |
| negative_control | cpu |  | 8 | 20 | 6 | no | 0.243 |  | PASS |
| schedule |  |  | 5 | 40 |  | yes | 0 |  | PASS |
| synthetic | cpu |  | 5 | 40 |  | yes | 0 | 0 | PASS |
| pipeline | cpu | float32 | 5 | 40 | 5 | yes | 0 | 0 | PASS |
| pipeline | cpu | bfloat16 | 5 | 40 | 5 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | cpu | bf16, stock latents bfloat16 | 5 | 40 | 5 | no | 0.19 |  | info |
| synthetic | mps |  | 5 | 40 |  | yes | 0 | 0 | PASS |
| pipeline | mps | float32 | 5 | 40 | 5 | yes | 0 | 0 | PASS |
| pipeline | mps | bfloat16 | 5 | 40 | 5 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | mps | bf16, stock latents bfloat16 | 5 | 40 | 5 | no | 0.189 |  | info |
| negative_control | cpu |  | 5 | 40 | 5 | no | 0.0842 |  | PASS |
| schedule |  |  | 5 | 20 |  | yes | 0 |  | PASS |
| synthetic | cpu |  | 5 | 20 |  | yes | 0 | 0 | PASS |
| pipeline | cpu | float32 | 5 | 20 | 5 | yes | 0 | 0 | PASS |
| pipeline | cpu | bfloat16 | 5 | 20 | 5 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | cpu | bf16, stock latents bfloat16 | 5 | 20 | 5 | no | 0.0783 |  | info |
| synthetic | mps |  | 5 | 20 |  | yes | 0 | 0 | PASS |
| pipeline | mps | float32 | 5 | 20 | 5 | yes | 0 | 0 | PASS |
| pipeline | mps | bfloat16 | 5 | 20 | 5 | yes | 0 | 0 | PASS |
| stock_bf16_deviation | mps | bf16, stock latents bfloat16 | 5 | 20 | 5 | no | 0.0958 |  | info |
| negative_control | cpu |  | 5 | 20 | 5 | no | 0.0956 |  | PASS |

`negative_control` passes when a deliberately wrong schedule (shift applied once) is detected.
`stock_bf16_deviation` compares the protocol's fp32 guidance and latents with the stock bf16 pipeline;
it is reported for information only.

## Recorded schedules

- shift 8, 40 steps, sigmas: 1.000000, 0.996748, 0.993344, 0.989778, 0.986038, 0.982111, 0.977982, 0.973634, 0.969052, 0.964215, 0.959100, 0.953684, 0.947939, 0.941834, 0.935334, 0.928400, 0.920987, 0.913043, 0.904509, 0.895316, 0.885386, 0.874626, 0.862927, 0.850161, 0.836176, 0.820787, 0.803771, 0.784858, 0.763710, 0.739907, 0.712915, 0.682046, 0.646403, 0.604783, 0.555547, 0.496392, 0.423986, 0.333315, 0.216471, 0.060207, 0.000000
- shift 8, 20 steps, sigmas: 1.000000, 0.993161, 0.985634, 0.977310, 0.968055, 0.957705, 0.946052, 0.932833, 0.917710, 0.900241, 0.879833, 0.855676, 0.826633, 0.791055, 0.746456, 0.688908, 0.611813, 0.503174, 0.338654, 0.060207, 0.000000
- shift 5, 40 steps, sigmas: 1.000000, 0.994791, 0.989360, 0.983694, 0.977775, 0.971587, 0.965111, 0.958327, 0.951212, 0.943742, 0.935888, 0.927620, 0.918906, 0.909707, 0.899983, 0.889686, 0.878766, 0.867163, 0.854811, 0.841635, 0.827550, 0.812460, 0.796251, 0.778795, 0.759942, 0.739518, 0.717317, 0.693098, 0.666571, 0.637391, 0.605138, 0.569301, 0.529245, 0.484181, 0.433106, 0.374732, 0.307372, 0.228781, 0.135892, 0.024414, 0.000000
- shift 5, 20 steps, sigmas: 1.000000, 0.989068, 0.977136, 0.964061, 0.949670, 0.933753, 0.916055, 0.896259, 0.873967, 0.848677, 0.819740, 0.786306, 0.747238, 0.700981, 0.645349, 0.577173, 0.491668, 0.381266, 0.233242, 0.024414, 0.000000
