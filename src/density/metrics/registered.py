"""Registered values for scoring (protocol v2 section 7; Milestone 5 decisions, approved by Chase on 2026-10-07).

Change any of these only through a dated amendment to the registration.

Every score is measured before lossy compression (protocol section 7; Chase, 2026-10-07, decisions 1 and 6).

Which machine computes what (approved plan, item 4):
- H100, inside each generation job: masked, outside-ROI and full-frame LPIPS, masked PSNR and SSIM, and crop LPIPS, all
  against the same seed's C1 reference, from both videos' lossless uint8 frames. Then the condition's lossless copy
  (FFV1, checked bit for bit) and the fixed CRF 10 MP4 (``density.video``), which is the playback copy.
- The 24 GB slice: the five VBench quality dimensions and the detector checks (face identity, hand detection, box
  overlap), from the lossless frames of the condition and its reference. The MP4s are scored too, as a descriptive
  check of the encoding's effect. A prompt-seed's reference and conditions are scored on one host.
- The Mac: nothing that produces a score; the pilot rules, the analysis and the dry runs.
"""

from torch.nn.attention import SDPBackend

# Videos.
FRAMES = 81
HEIGHT, WIDTH = 480, 832

# LPIPS (section 7): version 0.1 with AlexNet, from the verified data-only weight files (density.metrics.weights).
LPIPS_VERSION = "0.1"
LPIPS_NET = "alex"
LPIPS_INPUT = "RGB uint8 frames scaled to [-1, 1] as x / 127.5 - 1 (lpips.im2tensor), native 832 x 480, never resized"
MASKED_LPIPS = (
    "Equations 15 and 16: the spatial LPIPS map (spatial=True) of the complete frames, averaged inside the reference's "
    "measurement mask (the ROI dilated by 8 px) on each frame, then over the frames with a non-empty mask, each with "
    "equal weight. Pixels are never blacked out."
)
OUTSIDE_LPIPS = (
    "The same average on the complement of the reference ROI dilated by 32 px, over the frames where the complement is "
    "non-empty."
)
# Registered by Chase on 2026-10-07 (plan item 5): the HSA paper's metric, for the C2 check and the full-frame guardrail.
FULL_FRAME_LPIPS = "Scalar LPIPS (spatial=False) of each frame, averaged over all 81 frames with equal weight."
LPIPS_FRAMES_PER_BATCH = 9  # fixed, so no score depends on the memory available

# Descriptive pixel checks (plan item 6 and decision 9, registered by Chase on 2026-10-07), inside the measurement mask and over its frames.
MASKED_PSNR = (
    "Per frame, the mean squared error over the mask's pixels and the three channels; the mean over the mask's frames, "
    "each with equal weight; then 10 log10(255^2 / MSE), capped at 100 dB."
)
SSIM_K1, SSIM_K2, SSIM_DATA_RANGE = 0.01, 0.03, 255.0
SSIM_SIGMA, SSIM_TRUNCATE = 1.5, 3.5  # an 11 x 11 Gaussian window
MASKED_SSIM = (
    "Wang et al. 2004: an 11 x 11 Gaussian window with sigma 1.5, K1 0.01, K2 0.03, data range 255, population "
    "covariances, float64, on each RGB channel, with symmetric padding at the frame's edges (scikit-image's Gaussian "
    "SSIM). The channels' maps are averaged, the map is averaged inside the mask on each frame, and the frames are "
    "averaged with equal weight."
)
PSNR_CAP_DB = 100.0
CROP_MIN_SIDE = 64  # AlexNet's second max-pool leaves no features below 31 px
CROP_LPIPS = (
    "Scalar LPIPS of a crop of both frames: the bounding box of the frame's measurement mask, widened to at least 64 px "
    "on each side about its centre and kept inside the frame, the same box for the condition and the reference. The "
    "frames with a non-empty mask are averaged with equal weight."
)

# The five VBench quality dimensions (section 7), vendored from VBench v0.1.5 (third_party/README.md).
QUALITY_DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality")
# The 0-100 scale (plan item 2): VBench's per-video values, which are in [0, 1] for the first four and raw MUSIQ (0-100)
# for imaging quality. VBench divides only imaging quality's aggregate by 100; per-video MUSIQ values are kept as they are.
QUALITY_SCALE = {"subject_consistency": 100.0, "background_consistency": 100.0, "motion_smoothness": 100.0, "aesthetic_quality": 100.0, "imaging_quality": 1.0}
QUALITY_INPUT = "lossless frames"  # registered (decision 1): a reference's PNGs, a condition's FFV1 copy, checked by hash
QUALITY_MODE = (
    "Custom input: each dimension's per-video function from VBench v0.1.5 on every lossless frame of the video (a "
    "reference's PNGs, a condition's FFV1 copy, checked by hash); the video's CRF 10 MP4 is scored the same way as a "
    "descriptive check. Imaging quality uses VBench's default preprocessing ('longer': the longer side resized to 512). "
    "Motion smoothness runs at native resolution: the device is passed as a torch.device, so VBench's memory-based "
    "downscaling never applies; at 832 x 480 it would not apply on the H100 or the slice either. A non-finite feature or "
    "score stops the job (VBench would turn a NaN into 0)."
)
TEMPORAL_FLICKERING = "not a guardrail: it assumes static video (section 7)"

# Precision (plan item 10): fp32 with strict IEEE convolutions, except CLIP on CUDA, which runs in fp16 as VBench's
# clip.load does. On CUDA, scaled-dot-product attention (CLIP's) is pinned to the generation backends: flash, and
# memory-efficient where flash cannot run. DINO and MUSIQ compute attention with explicit matmuls; AMT has none.
CONV_FP32_PRECISION = "ieee"
MATMUL_FP32_PRECISION = "ieee"
ATTENTION_BACKENDS = (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION)  # in priority order
PRECISION = (
    "LPIPS, DINO, the LAION head, MUSIQ and AMT-S in fp32 with IEEE fp32 convolutions and matmuls (no TF32); CLIP in "
    "fp32 on CPU and fp16 on CUDA, as VBench runs it; CLIP's attention on CUDA through flash attention (memory-efficient "
    "where flash cannot run). Every scorer runs under the job's strict deterministic settings, and each switch is "
    "restored when scoring ends."
)

# Detector checks (plan item 7, decision 9 and section 7's descriptive checks; registered by Chase on 2026-10-07), with the
# mask pipeline's registered detectors and thresholds, on the lossless frames of the condition and its reference.
DETECTION_MATCH_IOU = 0.5  # a condition detection matches a reference detection at box IoU >= 0.5
FACE_IDENTITY = (
    "ArcFace R50 (buffalo_l w600k_r50) embeddings of faces SCRFD detects, aligned on SCRFD's five landmarks "
    "(insightface's norm_crop, 112 px). On every frame, each reference face is paired with the condition face of highest "
    "box IoU at IoU >= 0.5 (greedy); the frame's value is the mean cosine similarity of its pairs, and identity is the "
    "mean over frames with at least one pair. Face prompts only."
)
HAND_DETECTION = (
    "WiLoR hand boxes on every frame: a reference hand is detected again when a condition hand box (either side) overlaps "
    "it at IoU >= 0.5. Success is the share of reference hand boxes detected again; overlap is the mean of each reference "
    "box's best IoU (0 without any condition box). Manipulation prompts only."
)
BOX_OVERLAP = (
    "Each reference box's best IoU with a box of the same detector in the condition, averaged over every reference box on "
    "every frame: SCRFD faces for face prompts, WiLoR hands for manipulation, Mask R-CNN persons for full body."
)
DETECTORS = {"face": "scrfd", "manipulation": "wilor", "body": "mask_rcnn"}

# Scores. A missing or non-finite required score is a technical failure that blocks the decision (section 7):
# masked LPIPS and the seven guardrails. Descriptive checks record "not applicable" with a reason instead.
GUARDRAILS = ("full_frame_lpips", "outside_lpips", *QUALITY_DIMENSIONS)
REQUIRED = ("masked_lpips", *GUARDRAILS)
DESCRIPTIVE = ("masked_psnr_db", "masked_ssim", "crop_lpips", "face_identity", "hand_detection", "box_overlap",
               "quality_mp4")  # the last: the five dimensions on the MP4, and their difference from the lossless scores
LPIPS_LIMIT = 0.01  # section 9: full-frame and outside-ROI guardrail limits, in LPIPS
QUALITY_LIMIT = 0.5  # section 9: each quality dimension's limit, in points on the 0-100 scale

MACHINES = {
    "masked_lpips": "h100", "full_frame_lpips": "h100", "outside_lpips": "h100",
    "masked_psnr_db": "h100", "masked_ssim": "h100", "crop_lpips": "h100", "lossless_copy": "h100", "mp4_encode": "h100",
    **{d: "slice" for d in QUALITY_DIMENSIONS},
    "face_identity": "slice", "hand_detection": "slice", "box_overlap": "slice", "quality_mp4": "slice",
}
