# Mask pipeline

**Overall: PASS** (57/57 judged checks passed)

Code `75686b65401f11cbc55604a0b1c075e76d67bf51` (dirty: False); torch 2.14.1, macOS-26.3.1-arm64-arm-64bit.

| Test | Result |
| --- | --- |
| T19 RLE round trip is exact | PASS |
| T19 region masks are 81 x 480 x 832 bool | PASS |
| T20 three adjacent keyframes | PASS |
| T20 one keyframe is not enough | PASS |
| T20 gap breaks adjacency | PASS |
| T20 object must stay the same | PASS |
| T20 hand must stay the same | PASS |
| T20 trimmed when contact stops | PASS |
| T20 self contact does not count | PASS |
| T20 two separate runs | PASS |
| T20 ends at the last keyframe | PASS |
| T20 masks must be usable through the interval | PASS |
| T21 dilation radii are exact (disk lattice points) | PASS |
| T21 C is the dilated intersection on validated frames only | PASS |
| T22 rule fires exactly when IoU < 0.5 | PASS |
| T22 re-prompts land exactly where IoU first drops below 0.5 | PASS |
| T22 no re-prompt while SAM agrees with the detector | PASS |
| T22 hands crossing | PASS |
| T22 stray false positive | PASS |
| T22 side flip | PASS |
| T22 lost track | PASS |
| T22 termination | PASS |
| T22 hand tracks without a Hands23 confirmation are dropped | PASS |
| T22 confirmation counts only keyframes where the track has a mask | PASS |
| T22 confirmation ignores the side label and needs IoU >= 0.5 | PASS |
| T22 dropped tracks leave H and are not merged | PASS |
| T23 face ROI | PASS |
| T23 manipulation ROI | PASS |
| T23 body ROI | PASS |
| T23 A1 mask is H | C | PASS |
| T23 controls are an explicit empty mask, never detector-derived | PASS |
| T24 40 usable frames fail, 41 pass | PASS |
| T24 frames below 256 mask pixels are not usable | PASS |
| T24 manipulation needs a validated contact interval | PASS |
| T24 a frame outside the ROI dilated by 32 px is required | PASS |
| T24 one outside frame is enough | PASS |
| T24 both seeds must pass | PASS |
| T24 ordered replacement, stop when a reserve list runs out | PASS |
| T24 anatomy and image quality are never grounds for rejection | PASS |
| T25 inside = ROI dilated by 8 px, outside = not ROI dilated by 32 px | PASS |
| T25 frame sets from the reference | PASS |
| T25 identical, read-only and frozen across conditions | PASS |
| T28 extrapolated contact interval fails T20 | PASS |
| T28 body mask as the manipulation ROI fails T23 | PASS |
| T28 association today's ungated rule fails T22 | PASS |
| T28 association side-anchored without the IoU >= 0.5 override fails T22 | PASS |
| T28 association gate without the lost-track exception fails T22 | PASS |
| T28 association without the bound-detection rule fails T22 | PASS |
| T28 keeping every hand track fails T22 (hand-track confirmation) | PASS |
| Registered: every runtime weight file loads as data only | PASS |
| Conversion: WiLoR rebuilt from data-only files gives the original checkpoint's detections bitwise | PASS |
| T27 face clip: F usable in at least 41 frames | PASS |
| T27 hands clip: H usable in at least 41 frames and a validated contact interval | PASS |
| T27 body clip: B usable in at least 41 frames | PASS |
| T27 landscape clip: the control ROI is the explicit empty mask, and nothing is computed | PASS |
| T27 contact sheet for sign-off: face | info |
| T27 contact sheet for sign-off: hands | info |
| T27 contact sheet for sign-off: body | info |
| T27 contact sheet for sign-off: landscape | info |
| T26 repeated CPU run gives identical masks: face | PASS |
| T26 repeated CPU run gives identical masks: hands | PASS |
| T26 SAM 2.1 on MPS (detectors on CPU): face | info |

Association negative controls and the T22 cases each one fails:

- today's ungated rule: hands crossing, stray false positive, lost track
- side-anchored without the IoU >= 0.5 override: side flip
- gate without the lost-track exception: lost track
- without the bound-detection rule: termination

## Registered values

- Geometry 832 x 480, 81 frames at 16 fps. Keyframes every 8 frames.
- Eligibility: usable primary ROI (at least 256 mask pixels) in at least 41 of 81 frames; token coverage threshold 0.1.
- Dilations: contact 16 px, measurement 8 px, outside 32 px; structuring element: Euclidean disk: pixel offsets (dx, dy) with dx^2 + dy^2 <= r^2.
- Re-prompt when IoU < 0.5, only from a detection with IoU > 0.0; a track is lost at 0 mask pixels; Hands23 boxes join tracks at IoU >= 0.5; contact state `object_contact`.
- Association rule (2026-10-06): At each frame with detections, in order: (a) the detection that prompted a track at this frame belongs to it; (b) a detection that overlaps a track at IoU >= 0.5 belongs to that track whatever its side label (same-side track first when several qualify, then highest IoU, then lowest track ID); it never re-prompts or starts a track, and a different label is logged as a side disagreement; only a same-side detection confirms a track; (c) an unconfirmed track is re-prompted by the free same-side detection that overlaps it most at 0 < IoU < 0.5 (greedy by IoU, at most one re-prompt per track per frame); (d) an unconfirmed track whose mask has 0 pixels (lost) is re-prompted by a remaining free same-side detection, lowest track ID first; (e) any other detection starts a new track and fixes its side, unless it overlaps a track started at this frame at IoU >= 0.5. Sides apply to hand tracks (H) only; F, O and B have none. Frames are checked in order, and the clip is propagated again after the first frame where a detection re-prompts or starts a track; decisions are never undone, and tracks are never merged or deleted.
- Hands23 hand matching: A Hands23 hand box joins the hand track it overlaps at IoU >= 0.5, preferring the track with the same side, then the highest IoU; a different side label is logged as a side disagreement and does not drop the observation.
- Hand-track confirmation (2026-10-06): After association, a hand track is kept only if a Hands23 hand box of either side overlaps the track's mask box at IoU >= 0.5 on at least one keyframe within the track's span (the frames where the track has a mask, including those before its first detection). Unconfirmed tracks are dropped from H and never merged into another track; they stay in the metadata with the detection that started them and are outlined in grey on the contact sheets.
- Regions per stratum: Each reference computes only the regions its stratum uses: F for faces; H, O and C for manipulation; B for full body; nothing for controls, whose mask is the explicit empty mask.
- Frames before a track's first detection: SAM's masks before a track's first detection (backward propagation from its first prompt) are kept, flagged per track and frame in the metadata, drawn hatched on the contact sheets, and count toward the usable-frame rule; the audit judges them.
- Contact intervals require sustained contact: with keyframes 8 frames apart, contact shorter than 0.5 s cannot validate. The spacing is revisited only if the pilot's manipulation pass rate shows brief contacts are a common failure.
- SAM 2.1 precision: fp32 weights and activations, no autocast. SAM 2 stores memory features in bfloat16; a forward pre-hook on memory_attention casts memory and memory_pos to the dtype of the current features (fp32) on every frame, the same exact conversion PyTorch's type promotion already performs on frames with fp32 object pointers. JPEG frames at quality 100, 4:4:4.
- SAM 2.1 build: sam-2 built with SAM2_BUILD_CUDA=0 on both machines: the _C extension is absent, so the hole-filling post-processing (fill_holes_in_mask_scores) is skipped. A CUDA build on the pod would change the masks.
- Detector thresholds: SCRFD 0.5 at [640, 640] (onnxruntime CPU provider on both machines), WiLoR 0.3, Mask R-CNN person 0.5, Hands23 {'hand': 0.7, 'first_object': 0.5, 'second_object': 0.3, 'hand_relation': 0.3, 'object_relation': 0.7}.
- Every runtime weight file loads as data only (safetensors, ONNX, or torch.load(weights_only=True)).

## Clips

| Clip | Stratum | Source | License | Segment | Source frames | Frames SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| face | face | [20140529 - Va et vient.webm](https://commons.wikimedia.org/wiki/File:20140529_-_Va_et_vient.webm), Clément Bucco-Lechat | CC BY-SA 3.0 | 2 s | 60-210 (2.000-7.000 s) | `3ede7b573a10c66d` |
| hands | manipulation | [Video of a complete use session with a gyroscopic exercise tool.webm](https://commons.wikimedia.org/wiki/File:Video_of_a_complete_use_session_with_a_gyroscopic_exercise_tool.webm), Pittigrilli | CC BY-SA 4.0 | 70 s | 1750-1875 (70.003-75.003 s) | `cff42ec42f6684f4` |
| body | body | [Kettlebell juggling flow by Taco Fleur.webm](https://commons.wikimedia.org/wiki/File:Kettlebell_juggling_flow_by_Taco_Fleur.webm), Taco Fleur | CC BY-SA 4.0 | 19 s | 570-720 (19.003-24.003 s) | `9120ec3819d12aaf` |
| landscape | control | [South Fork Clackamas Wild and Scenic River (28287735390).webm](https://commons.wikimedia.org/wiki/File:South_Fork_Clackamas_Wild_and_Scenic_River_(28287735390).webm), BLM Oregon & Washington | Public domain (US federal work, PD-USGov-BLM); also CC BY 2.0 on Flickr | 1 s | 29-179 (1.001-6.006 s) | `5d595e63715f56e9` |

Conformed with ffmpeg version 8.0.1 Copyright (c) 2000-2025 the FFmpeg developers: the nearest source frame to each 1/16 s step (no interpolation), then `scale=-1:480:flags=area,crop=832:480`.

## Results per clip

| Clip | Stratum | Tracks (side, first detection) | Re-prompts | New tracks after frame 0 | Side disagreements | Frames before first detection | ROI usable | Outside frames | Eligibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face | face | F0 (-, 0) | 0 | 0 | 0 | 0 | 81 | 81 | pass |
| hands | manipulation | H0 (right, 0) | 0 | 0 | 0 | 0 | 81 | 81 | pass |
| hands | manipulation | O0 (-, 0) | 1 | 0 | 0 | 0 |  | | |
| body | body | B0 (-, 0), B1 (-, 0), B2 (-, 24), B3 (-, 51) | 1 | 2 | 0 | 52 | 81 | 81 | not gated |
| body | body (contact test) | H0 (left, 0), H1 (right, 29), H2 (right, 38, dropped), H3 (right, 52, dropped), H4 (right, 70, dropped) | 4 | 4 | 6 | 163 |  | | |
| body | body (contact test) | O0 (-, 32) | 0 | 1 | 0 | 32 |  | | |
| landscape | control | nothing computed (explicit empty mask) | | | | | 0 | 81 | not gated |

### hands: Hands23 contact (manipulation reference)

| Keyframe | Hands23 hands: side, score -> hand track: contact state, object track |
| --- | --- |
| 0 | R 1.00 -> H0: object_contact O0 |
| 8 | R 1.00 -> H0: object_contact O0 |
| 16 | R 1.00 -> H0: object_contact O0 |
| 24 | R 1.00 -> H0: object_contact O0 |
| 32 | R 1.00 -> H0: object_contact O0 |
| 40 | R 1.00 -> H0: object_contact O0 |
| 48 | R 1.00 -> H0: object_contact O0 |
| 56 | R 1.00 -> H0: object_contact O0 |
| 64 | R 1.00 -> H0: object_contact O0 |
| 72 | R 1.00 -> H0: object_contact O0 |
| 80 | R 1.00 -> H0: object_contact O0 |

Hand tracks confirmed by a Hands23 hand box: H0 at keyframes 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80. Dropped without confirmation: none.

Contact intervals: H0-O0 frames 0-80. Hands23 side disagreements with the hand tracks: 0.

### body: Hands23 contact (informational contact-transition test)

| Keyframe | Hands23 hands: side, score -> hand track: contact state, object track |
| --- | --- |
| 0 | L 0.99 -> H0: no_contact |
| 8 | L 1.00 -> H0: no_contact |
| 16 | L 1.00 -> H0: no_contact |
| 24 | L 1.00 -> H0: no_contact; R 0.90 -> untracked: no_contact |
| 32 | L 1.00 -> H0: no_contact; R 0.77 -> H1: object_contact O0 |
| 40 | L 1.00 -> H0: no_contact |
| 48 | L 1.00 -> H0: no_contact |
| 56 | L 1.00 -> H0: no_contact; R 0.96 -> untracked: no_contact |
| 64 | L 1.00 -> H0: no_contact |
| 72 | L 1.00 -> H0: no_contact |
| 80 | R 0.99 -> H1: object_contact O0 |

Hand tracks confirmed by a Hands23 hand box: H0 at keyframes 0, 8, 16, 24, 32, 40, 48, 56, 64, 72; H1 at keyframes 32, 80. Dropped without confirmation: H2 (started at frame 38), H3 (started at frame 52), H4 (started at frame 70).

Contact intervals: none. Hands23 side disagreements with the hand tracks: 0.
No interval validates: contact must be seen at two adjacent keyframes (0.5 s apart) with the same hand and object tracks, and brief catches between keyframes cannot satisfy that (registered as sustained contact).

## Timings (Mac CPU, per 81-frame clip)

| Clip | Component seconds | SAM calls |
| --- | --- | --- |
| face | total 59.2, roi_coverage_measurement_eligibility 0.9, F.detector 4.9, F.sam 51.4 | 1 |
| hands | total 175.3, roi_coverage_measurement_eligibility 0.9, H.detector 4.5, H.sam 51.1, O.hands23 9.7, O.sam 105.2 | 3 |
| body | total 1755.5, roi_coverage_measurement_eligibility 0.9, B.detector 31.8, B.sam 459.6, H.detector 4.7, H.sam 1192.5, O.hands23 9.9, O.sam 51.5 | 14 |
| landscape | total 0, roi_coverage_measurement_eligibility 0 | 0 |

Model builds (once per session): scrfd 0 s, wilor 0.2 s, hands23 1.2 s, mask_rcnn 0.4 s. The manipulation path on the two-hand contact-test clip took 1258.5 s (body).

| Stratum | References | Seconds per reference | Hours | Note |
| --- | --- | --- | --- | --- |
| face | 84 | 59.2 | 1.38 |  |
| manipulation | 84 | 175.3-1258.5 | 4.09-29.36 | low: the one-hand gyroscope clip; high: the manipulation path on the two-hand kettlebell clip |
| body | 16 | 492.3 | 2.19 |  |
| control | 16 | 0 | 0 | nothing computed: the explicit empty mask (0.0 s for the ROI, coverage and measurement masks) |

About 7.66-32.93 hours for the masks of about 200 references at Mac CPU speed (Apple M4 Pro CPU, one clip at a time; the pod's GPU timings must be measured there), depending on how many manipulation references look like the two-hand clip. SAM 2.1 dominates: every association round propagates every track again, so each extra track, re-prompt or new track adds a full pass, and passes slow down as tracks accumulate.

## T26: repeatability

- T26 repeated CPU run gives identical masks: face: PASS
- T26 repeated CPU run gives identical masks: hands: PASS
- SAM 2.1 on MPS, face clip (detectors on CPU): two MPS runs bitwise equal: True (0 differing pixels); against CPU, 5 differing pixels, per-frame mask IoU min 0.999985, mean 0.9999991; same re-prompt events: True. SAM time [50.4, 50.3] s on MPS vs 51.4 s on CPU.

## Versions and weights

- torch: 2.14.1
- torchvision: 0.29.1
- sam-2: 1.0 (git 2b90b9f5ceec907a1c18123530e92e794ad901a4)
- detectron2: 0.6 (git 1e3e13bbf607b54f62205c4c33922521822fb298)
- insightface: 2.1
- onnxruntime: 1.23.2
- onnx: 1.23.1
- ultralytics: 8.4.123
- hydra-core: 1.3.7
- iopath: 0.1.10
- fvcore: 0.1.5.post20221221
- scipy: 1.15.3
- scikit-image: 0.25.2
- opencv-python: 4.14.0.94
- pillow: 12.3.0
- safetensors: 0.8.0
- numpy: 2.2.6
- ffprobe version 8.0.1 Copyright (c) 2007-2025 the FFmpeg developers
- Hands23 vendored at 1bc0f919ffa7a9f375e7e8042e1d26f3743d5819, patch `third_party/patches/hands23_detector.diff` (SHA-256 `64963832910c092f`)

| Weight file | Size | SHA-256 | License |
| --- | --- | --- | --- |
| sam2.1_hiera_small | 184,416,285 | `6d1aa6f30de5c922` | Apache-2.0 |
| scrfd_10g | 16,923,827 | `5838f7fe053675b1` | InsightFace pretrained models: non-commercial research only |
| wilor_detector | 53,582,271 | `5ef3df44e42d2db5` | CC BY-NC-ND 4.0 (WiLoR license.txt); loaded by Ultralytics (AGPL-3.0) |
| wilor_detector_safetensors | 53,378,530 | `2fd84d6d39167c11` | as wilor_detector |
| wilor_detector_meta | 3,960 | `9baef4db12875fd1` | as wilor_detector |
| mask_rcnn_r50_fpn_3x | 177,841,981 | `9a737e290372f1f7` | CC BY-SA 3.0 (Detectron2 Model Zoo) |
| mask_rcnn_r50_fpn_3x_safetensors | 177,832,424 | `13451037b8350c23` | as mask_rcnn_r50_fpn_3x |
| hands23_original | 467,798,701 | `b3d2ff966d8a19b9` | not stated by the authors (code is MIT) |
| hands23 | 467,652,176 | `4051f3db3509c9a5` | as hands23_original |
| wilor_conversion_reference | 9,140 | `b3bb29fa872ca20d` | as wilor_detector |
| wilor_conversion_image_AR_9HvbAaq670Y_19_3510_5 | 261,206 | `a880b33c1c8f2430` | Hands23 demo image; its source dataset's terms apply; not redistributed |
| wilor_conversion_image_CC_000000057750 | 236,322 | `5df6d24ed2c81d0c` | Hands23 demo image; its source dataset's terms apply; not redistributed |
| wilor_conversion_image_EK_0010_P04_101_frame_0000013113 | 595,115 | `b2dfabc4d22dc1e4` | Hands23 demo image; its source dataset's terms apply; not redistributed |
| wilor_conversion_image_EK_0028_P12_101_frame_0000086975 | 395,961 | `4161cc2c88040c0a` | Hands23 demo image; its source dataset's terms apply; not redistributed |
| wilor_conversion_image_ND_nQLywBlIF7I_frame005401 | 136,120 | `fb50d1a4d09e6666` | Hands23 demo image; its source dataset's terms apply; not redistributed |

## Contact sheets and audit

Sheets are not versioned; their hashes are in the JSON and `mask_pipeline.sha256`.

- `results/checks/mask_pipeline/face.png` `19ded3167b1ea068`
- `results/checks/mask_pipeline/hands.png` `bdf0ea96a4a66732`
- `results/checks/mask_pipeline/body.png` `0066c2ffa1c10ecf`
- `results/checks/mask_pipeline/body_contact_test.png` `3fb1bb30c130caa6`
- `results/checks/mask_pipeline/landscape.png` `1169e868780a4018`

Reviewer sign-off goes in `results/checks/mask_pipeline_audit.csv` (accept or reject; a rejection names one of the permitted audit reasons). A rerun keeps a reviewer's entries only for sheets whose hash is unchanged.
