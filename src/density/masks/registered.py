"""Registered values for the mask pipeline (protocol v2 sections 4 and 7; Milestone 4 decisions).

Change any of these only through a dated amendment to the registration.
"""

from density.allocation import ROI_THRESHOLD  # noqa: F401  (b = 1[m >= 0.1], shared with allocation)

# Reference video geometry.
FRAMES = 81
HEIGHT = 480
WIDTH = 832
FPS = 16

# Eligibility (section 4).
MIN_USABLE_FRAMES = 41  # a usable primary ROI in at least 41 of the 81 frames
MIN_USABLE_PIXELS = 256  # a usable ROI frame has at least one token cell's worth of mask pixels

# Detection, propagation and contact.
KEYFRAME_SPACING = 8  # Hands23 keyframes: frames 0, 8, ..., 80
REPROMPT_IOU = 0.5  # re-prompt SAM when its IoU with a fresh detection is below 0.5 (strict)
REPROMPT_GATE_IOU = 0.0  # ... but only from a detection that overlaps the track at all (IoU > 0, strict)
LOST_TRACK_PIXELS = 0  # a track is lost at a frame where its mask has this many pixels
OBJECT_MATCH_IOU = 0.5  # a Hands23 hand or object box joins a SAM track at IoU >= 0.5
CONTACT_STATE = "object_contact"  # the only Hands23 contact state that counts as contact
HAND_SIDES = ("left", "right")  # WiLoR classes 0/1 and Hands23 left_hand/right_hand, in one vocabulary

# Association of detections with SAM tracks (Milestone 4 decision, 2026-10-06), implemented in tracking.Association.
ASSOCIATION_RULE = (
    "At each frame with detections, in order: "
    "(a) the detection that prompted a track at this frame belongs to it; "
    "(b) a detection that overlaps a track at IoU >= 0.5 belongs to that track whatever its side label "
    "(same-side track first when several qualify, then highest IoU, then lowest track ID); it never re-prompts or starts a "
    "track, and a different label is logged as a side disagreement; only a same-side detection confirms a track; "
    "(c) an unconfirmed track is re-prompted by the free same-side detection that overlaps it most at 0 < IoU < 0.5 "
    "(greedy by IoU, at most one re-prompt per track per frame); "
    "(d) an unconfirmed track whose mask has 0 pixels (lost) is re-prompted by a remaining free same-side detection, "
    "lowest track ID first; "
    "(e) any other detection starts a new track and fixes its side, unless it overlaps a track started at this frame at "
    "IoU >= 0.5. Sides apply to hand tracks (H) only; F, O and B have none. "
    "Frames are checked in order, and the clip is propagated again after the first frame where a detection re-prompts "
    "or starts a track; decisions are never undone, and tracks are never merged or deleted."
)
HANDS23_HAND_MATCH = (
    "A Hands23 hand box joins the hand track it overlaps at IoU >= 0.5, preferring the track with the same side, then the "
    "highest IoU; a different side label is logged as a side disagreement and does not drop the observation."
)
# Hand-track confirmation (Milestone 4 decision, 2026-10-06), applied after association in regions.confirm_hand_tracks.
HAND_CONFIRM_IOU = 0.5  # a Hands23 hand box confirms a hand track at IoU >= 0.5 with the track's mask box
HAND_TRACK_CONFIRMATION = (
    "After association, a hand track is kept only if a Hands23 hand box of either side overlaps the track's mask box at "
    "IoU >= 0.5 on at least one keyframe within the track's span (the frames where the track has a mask, including those "
    "before its first detection). Unconfirmed tracks are dropped from H and never merged into another track; they stay in "
    "the metadata with the detection that started them and are outlined in grey on the contact sheets."
)
STRATUM_REGIONS = (
    "Each reference computes only the regions its stratum uses: F for faces; H, O and C for manipulation; B for full body; "
    "nothing for controls, whose mask is the explicit empty mask."
)
BACKFILL = (
    "SAM's masks before a track's first detection (backward propagation from its first prompt) are kept, flagged per track "
    "and frame in the metadata, drawn hatched on the contact sheets, and count toward the usable-frame rule; the audit "
    "judges them."
)

# Contact must be sustained. An interval needs object_contact at two adjacent keyframes, which are 8 frames (0.5 s at
# 16 fps) apart, so contact that lasts 8 frames or fewer can never validate, and contact that lasts at least 16 frames
# always spans two keyframes. Revisit the spacing only if the pilot's manipulation pass rate shows that brief contacts
# are a common failure.
MIN_CONTACT_SECONDS = KEYFRAME_SPACING / FPS
SUSTAINED_CONTACT = (
    f"Contact intervals require sustained contact: with keyframes {KEYFRAME_SPACING} frames apart, contact shorter than "
    f"{MIN_CONTACT_SECONDS:g} s cannot validate. The spacing is revisited only if the pilot's manipulation pass rate shows "
    "brief contacts are a common failure."
)

# Dilations, all with one structuring element.
CONTACT_DILATION_PX = 16  # C = dilate(H, 16) & dilate(O, 16), equation 9
MEASURE_DILATION_PX = 8  # measurement mask: ROI dilated by 8 px (section 7)
OUTSIDE_DILATION_PX = 32  # outside mask: complement of the ROI dilated by 32 px (section 7)
STRUCTURING_ELEMENT = "Euclidean disk: pixel offsets (dx, dy) with dx^2 + dy^2 <= r^2"

# SAM 2.1 reads JPEG frame folders only; frames are written at this quality with 4:4:4 chroma.
SAM_JPEG_QUALITY = 100
# SAM 2.1 numerics, identical on the Mac and the pod (Milestone 4 decision, 2026-10-06).
SAM_PRECISION = (
    "fp32 weights and activations, no autocast. SAM 2 stores memory features in bfloat16; a forward pre-hook on "
    "memory_attention casts memory and memory_pos to the dtype of the current features (fp32) on every frame, the same "
    "exact conversion PyTorch's type promotion already performs on frames with fp32 object pointers."
)
# SAM 2.1's fp32 convolutions on CUDA (2026-10-06, smoke test): strict IEEE fp32 rather than cuDNN's TF32 default,
# by Chase's rule (strict fp32 unless it adds more than about 10 percent to per-reference mask time). On the 24 GB slice,
# strict fp32 for the whole pipeline took 96.8 s against 98.7 s (face) and 520.1 s against 515.9 s (manipulation, +0.8
# percent); SAM's own time went from 10.2 to 9.6 s and from 335.0 to 336.7 s. Masks moved by IoU 0.99972 to 0.99999.
# Applies to SAM only; the detectors keep the default. No effect on CPU or MPS.
SAM_CONV_FP32_PRECISION = "ieee"
SAM_BUILD = (
    "sam-2 built with SAM2_BUILD_CUDA=0 on both machines: the _C extension is absent, so the hole-filling post-processing "
    "(fill_holes_in_mask_scores) is skipped. A CUDA build on the pod would change the masks."
)
# Detectron2's ROIAlign (Mask R-CNN and Hands23) on CUDA (2026-10-07, Chase; smoke test Session E): torchvision's
# kernel for inference, not the pure-Python version torchvision substitutes under deterministic algorithms for a
# deterministic backward pass (about 20 s an image on the slice). The kernel's forward pass has no atomics; repeats were
# bitwise, outputs differed only in rounding (boxes by at most 0.00012 px), and the smoke-1 and smoke-2 masks were identical.
DETECTRON2_ROI_ALIGN = (
    "torchvision's roi_align kernel (torch.ops.torchvision.roi_align) for every Detectron2 ROIAlign, inference only, with a "
    "check on every Mask R-CNN and Hands23 call that it ran (density/masks/roi_align.py)"
)
# Where the experiment's masks are computed (Milestone 4 decision, 2026-10-06).
MASK_GPU = (
    "The experiment's reference masks are computed on one GPU type: the 24 GB MIG slice of an NVIDIA RTX PRO 6000 "
    "Blackwell Server Edition (RunPod Secure Cloud, US-NE-1, \"NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb\"). "
    "SAM 2.1, WiLoR, Hands23 and Mask R-CNN run on it; SCRFD stays on the onnxruntime CPU provider. The mask "
    "determinism check (T26) runs on the same GPU. Video generation runs on an H100 SXM, never on this slice."
)

# Detector score thresholds.
SCRFD_THRESHOLD = 0.5
SCRFD_INPUT_SIZE = (640, 640)  # InsightFace's default detector input size
WILOR_THRESHOLD = 0.3  # WiLoR demo default
MASK_RCNN_PERSON_THRESHOLD = 0.5
HANDS23_THRESHOLDS = {  # Hands23 demo defaults
    "hand": 0.7,
    "first_object": 0.5,
    "second_object": 0.3,
    "hand_relation": 0.3,
    "object_relation": 0.7,
}
