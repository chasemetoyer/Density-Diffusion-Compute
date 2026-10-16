"""Unit tests for the slice mask-speed diagnosis (Session E): the opt-in SCRFD thread count and the output digest."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MASK_WEIGHTS = (ROOT / "weights" / "masks").exists()


@pytest.mark.skipif(not MASK_WEIGHTS, reason="mask-pipeline weights not downloaded")
def test_scrfd_thread_count_is_opt_in():
    from density.masks.detectors import SCRFDFaceDetector

    default = SCRFDFaceDetector().model.session.get_session_options()
    assert default.intra_op_num_threads == 0  # onnxruntime's own default
    capped = SCRFDFaceDetector(threads=2).model.session.get_session_options()
    assert (capped.intra_op_num_threads, capped.inter_op_num_threads) == (2, 1)


def test_diagnosis_digest_is_bitwise():
    spec = importlib.util.spec_from_file_location("mask_threads", ROOT / "scripts" / "pod" / "diagnostics" / "mask_threads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = [{"box": np.array([1.0, 2.0], dtype=np.float32), "score": 0.5}]
    assert module.digest(a) == module.digest([{"score": 0.5, "box": np.array([1.0, 2.0], dtype=np.float32)}])
    assert module.digest(a) != module.digest([{"box": np.nextafter(a[0]["box"], np.float32(3)), "score": 0.5}])
    assert module.digest(a) != module.digest([{"box": a[0]["box"].astype(np.float64), "score": 0.5}])
