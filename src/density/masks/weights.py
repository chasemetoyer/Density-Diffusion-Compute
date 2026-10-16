"""Mask-pipeline weight files: sources, licenses and SHA-256, verified before anything loads them.

Files live under ``weights/masks/`` (not versioned). Nothing in the pipeline
downloads on first use; a missing or altered file is an error. Every file
listed here is copied to the pod with the weights, including the WiLoR
conversion-check reference under ``wilor/conversion_check/``.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "weights" / "masks"


@dataclass(frozen=True)
class WeightFile:
    name: str
    path: str  # relative to ROOT
    source: str
    size: int
    sha256: str
    sha256_source: str
    license: str


WEIGHTS = {
    w.name: w
    for w in (
        WeightFile(
            "sam2.1_hiera_small",
            "sam2.1-hiera-small/sam2.1_hiera_small.pt",
            "huggingface.co/facebook/sam2.1-hiera-small @ ee5bba1d82bb8749febdf90f45e84b687142ba03",
            184416285,
            "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
            "Hugging Face LFS (Meta publishes none)",
            "Apache-2.0",
        ),
        WeightFile(
            "scrfd_10g",
            "insightface/models/buffalo_l/det_10g.onnx",
            "github.com/deepinsight/insightface release model-zoo, buffalo_l.zip "
            "(sha256 80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f, official release digest)",
            16923827,
            "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
            "computed from the verified zip; matches two third-party mirrors",
            "InsightFace pretrained models: non-commercial research only",
        ),
        WeightFile(
            "wilor_detector",
            "wilor/pretrained_models/detector.pt",
            "huggingface.co/spaces/rolpotamias/WiLoR @ 99fe3d7acff8104ecca1055df7467709506c2fa6",
            53582271,
            "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
            "Hugging Face LFS",
            "CC BY-NC-ND 4.0 (WiLoR license.txt); loaded by Ultralytics (AGPL-3.0)",
        ),
        WeightFile(
            "wilor_detector_safetensors",
            "wilor/pretrained_models/detector.safetensors",
            "converted from wilor_detector by scripts/convert_wilor_detector.py in a throwaway environment "
            "(ultralytics 8.4.123, dill 0.4.1); state dict as stored (fp16 weights, int64 BatchNorm counters)",
            53378530,
            "2fd84d6d39167c11f45d05989ae7d39d98eab786de25852166ada44f8cab1d4c",
            "computed at conversion",
            "as wilor_detector",
        ),
        WeightFile(
            "wilor_detector_meta",
            "wilor/pretrained_models/detector.json",
            "written by scripts/convert_wilor_detector.py: architecture dict, names, keypoint shape, stride, "
            "activation overrides (SPPF cv1 keeps the checkpoint's SiLU), checkpoint prediction arguments, provenance",
            3960,
            "9baef4db12875fd19612632bfb42fa16de1173b961bd11ebaacad9a79d1dd46a",
            "computed at conversion",
            "as wilor_detector",
        ),
        WeightFile(
            "mask_rcnn_r50_fpn_3x",
            "detectron2/model_final_f10217.pkl",
            "dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
            177841981,
            "9a737e290372f1f70994ebcbd89d8004dbb3ae30a605fd915a190fa4a782dd66",
            "recorded at download after its md5 f10217d075016156e04f19242e20d8d8 matched the S3 metadata",
            "CC BY-SA 3.0 (Detectron2 Model Zoo)",
        ),
        WeightFile(
            "mask_rcnn_r50_fpn_3x_safetensors",
            "detectron2/model_final_f10217.safetensors",
            "converted from mask_rcnn_r50_fpn_3x by scripts/convert_mask_rcnn.py (restricted unpickler, four globals); "
            "drops only the five stale anchor_generator.cell_anchors keys Detectron2 ignores",
            177832424,
            "13451037b8350c23645bad6c689cf49e0fc9b1383a26caa2eb322c90121c53bd",
            "computed at conversion; outputs bitwise identical to the original on the Hands23 demo images",
            "as mask_rcnn_r50_fpn_3x",
        ),
        WeightFile(
            "hands23_original",
            "hands23/model_hands23.pth",
            "fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth (expired TLS certificate)",
            467798701,
            "b3d2ff966d8a19b991a3ce170e2d8f86040f614ef2edf4a55423288b278f723c",
            "trust on first use: recorded at download, not authenticated",
            "not stated by the authors (code is MIT)",
        ),
        WeightFile(
            "hands23",
            "hands23/model_hands23.safetensors",
            "converted from hands23_original with torch.load(weights_only=True); the only Hands23 file the pipeline loads",
            467652176,
            "4051f3db3509c9a5d8d6074fee1b48ba564c09a34a117a9711825ff4d12f116f",
            "computed at conversion",
            "as hands23_original",
        ),
        WeightFile(
            "wilor_conversion_reference",
            "wilor/conversion_check/reference.npz",
            "written by scripts/convert_wilor_detector.py: the original detector.pt's detections (YOLO.predict, conf 0.3, "
            "CPU) on the five images below, which density.checks.masks compares the rebuilt detector against",
            9140,
            "b3bb29fa872ca20d49e8a044b1f1d56c22c384ee326554c8e57e8e7252945c57",
            "computed at conversion",
            "as wilor_detector",
        ),
        WeightFile(
            "wilor_conversion_image_AR_9HvbAaq670Y_19_3510_5",
            "wilor/conversion_check/images/AR_9HvbAaq670Y_19_3510_5.jpg",
            "Hands23 demo image demo_example/example_images/AR_9HvbAaq670Y_19_3510_5.jpg (github.com/EvaCheng-cty/hands23_detector @ 1bc0f91)",
            261206,
            "a880b33c1c8f2430dd2b526d34975b83c5c7a24282281577019bcd610cbbad6e",
            "computed when copied from the Hands23 repository",
            "Hands23 demo image; its source dataset's terms apply; not redistributed",
        ),
        WeightFile(
            "wilor_conversion_image_CC_000000057750",
            "wilor/conversion_check/images/CC_000000057750.jpg",
            "Hands23 demo image demo_example/example_images/CC_000000057750.jpg (github.com/EvaCheng-cty/hands23_detector @ 1bc0f91)",
            236322,
            "5df6d24ed2c81d0c3839cd7f0b64163c3038f79fd28002c57e5e00b39b5e90a5",
            "computed when copied from the Hands23 repository",
            "Hands23 demo image; its source dataset's terms apply; not redistributed",
        ),
        WeightFile(
            "wilor_conversion_image_EK_0010_P04_101_frame_0000013113",
            "wilor/conversion_check/images/EK_0010_P04_101_frame_0000013113.jpg",
            "Hands23 demo image demo_example/example_images/EK_0010_P04_101_frame_0000013113.jpg (github.com/EvaCheng-cty/hands23_detector @ 1bc0f91)",
            595115,
            "b2dfabc4d22dc1e430ec6d58cd660cd5584b5d9522511c01528d29b3ec446ff5",
            "computed when copied from the Hands23 repository",
            "Hands23 demo image; its source dataset's terms apply; not redistributed",
        ),
        WeightFile(
            "wilor_conversion_image_EK_0028_P12_101_frame_0000086975",
            "wilor/conversion_check/images/EK_0028_P12_101_frame_0000086975.jpg",
            "Hands23 demo image demo_example/example_images/EK_0028_P12_101_frame_0000086975.jpg (github.com/EvaCheng-cty/hands23_detector @ 1bc0f91)",
            395961,
            "4161cc2c88040c0ab3d7fb44d983a0c55cb01de496dce53a60859709924c9aba",
            "computed when copied from the Hands23 repository",
            "Hands23 demo image; its source dataset's terms apply; not redistributed",
        ),
        WeightFile(
            "wilor_conversion_image_ND_nQLywBlIF7I_frame005401",
            "wilor/conversion_check/images/ND_nQLywBlIF7I_frame005401.jpg",
            "Hands23 demo image demo_example/example_images/ND_nQLywBlIF7I_frame005401.jpg (github.com/EvaCheng-cty/hands23_detector @ 1bc0f91)",
            136120,
            "fb50d1a4d09e66661a869ef028ce8b421da16ef6c702db7f3be7be3dfca61caf",
            "computed when copied from the Hands23 repository",
            "Hands23 demo image; its source dataset's terms apply; not redistributed",
        ),
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(name: str) -> Path:
    """The weight file's path, after checking its size and SHA-256."""
    weight = WEIGHTS[name]
    path = ROOT / weight.path
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is missing; weights are never downloaded at run time")
    if path.stat().st_size != weight.size or sha256_file(path) != weight.sha256:
        raise RuntimeError(f"{name}: {path} does not match its registered size and SHA-256")
    return path
