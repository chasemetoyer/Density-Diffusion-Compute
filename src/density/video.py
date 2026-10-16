"""The registered video files (Chase, 2026-10-07): a fixed CRF 10 MP4 for playback and a lossless copy for scoring.

Every score is measured before lossy compression (protocol section 7): LPIPS and the pixel checks inside each generation
job, and the quality dimensions and detector checks from the lossless copy. References keep their lossless frames as
PNG; conditions keep theirs as FFV1 (``encode_lossless``), each checked to decode bit for bit to the frames it stores.
The MP4 is the playback copy, and its quality scores are recorded beside the lossless ones as a descriptive check.

The MP4:

- Encoder: PyAV 17.1.0's bundled FFmpeg 8.1.1 and x264 (commit b35605a); libx264, CRF 10, preset medium, yuv420p,
  16 fps; one thread and ``cpu-independent=1``, so the bitstream does not depend on the CPU or thread count.
- Colour: RGB to BT.709 limited-range YCbCr in numpy (round half to even), with each 2x2 block of chroma averaged.
  The decoder inverts it in numpy, repeating each chroma sample over its 2x2 block. swscale is never used, so the
  conversion is the same on every machine. H.264 decoding is bit-exact by specification.
- Compare decoded frames (``frames_sha256``), never file hashes: the container carries version strings.
- Every encode records the decoded MP4's PSNR against its lossless frames.

The lossless copy: FFV1 in Matroska, pixel format bgra, one thread. RGB becomes BGRA (alpha 255) and back by reordering
channels in numpy, so no colour conversion is involved; the decoded file must equal the stored frames bit for bit.
"""

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np

CODEC = "libx264"
CRF = 10
PRESET = "medium"
PIX_FMT = "yuv420p"
FPS = 16
X264_PARAMS = "threads=1:cpu-independent=1:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"  # colour tags for players;
# they do not change decoded pixels, which are converted in numpy
KR, KB = 0.2126, 0.0722  # BT.709
KG = 1.0 - KR - KB
REQUIRED_OPTIONS = {"crf": f"{CRF:.1f}", "threads": "1"}  # checked in every file's x264 settings string
ENCODING = {
    "codec": CODEC, "crf": CRF, "preset": PRESET, "pix_fmt": PIX_FMT, "fps": FPS, "x264_params": X264_PARAMS,
    "colour": "BT.709 limited range, converted in numpy (round half to even); chroma: 2x2 mean when encoding, 2x2 repeat when decoding",
    "library": "PyAV 17.1.0 with its bundled FFmpeg 8.1.1 and x264 (core 165, commit b35605a)",
}
VERSIONS = {"pyav": "17.1.0", "ffmpeg": "8.1.1", "x264_core": "165"}  # checked on every encode
LOSSLESS = {"codec": "ffv1", "container": "matroska", "pix_fmt": "bgra", "threads": 1, "fps": FPS,
            "conversion": "RGB to BGRA (alpha 255) and back by reordering channels in numpy; no colour conversion"}


def rgb_to_yuv420(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB to (T, 3H/2, W) uint8 I420 planes (Y, then U, then V), BT.709 limited range."""
    t, h, w, c = frames.shape
    if c != 3 or h % 2 or w % 2 or frames.dtype != np.uint8:
        raise ValueError(f"expected (T, even H, even W, 3) uint8 frames, got {frames.shape} {frames.dtype}")
    x = frames.astype(np.float64) / 255.0
    y = KR * x[..., 0] + KG * x[..., 1] + KB * x[..., 2]
    cb = (x[..., 2] - y) / (2.0 * (1.0 - KB))
    cr = (x[..., 0] - y) / (2.0 * (1.0 - KR))
    pool = lambda a: ((a[:, 0::2, 0::2] + a[:, 0::2, 1::2]) + (a[:, 1::2, 0::2] + a[:, 1::2, 1::2])) / 4.0  # fixed order, so no reduction path changes the rounding
    Y = np.clip(np.round(16.0 + 219.0 * y), 16, 235).astype(np.uint8)
    U = np.clip(np.round(128.0 + 224.0 * pool(cb)), 16, 240).astype(np.uint8)
    V = np.clip(np.round(128.0 + 224.0 * pool(cr)), 16, 240).astype(np.uint8)
    return np.concatenate([Y.reshape(t, -1), U.reshape(t, -1), V.reshape(t, -1)], axis=1).reshape(t, h * 3 // 2, w)  # PyAV's flat I420 layout


def yuv420_to_rgb(planes: np.ndarray) -> np.ndarray:
    """The inverse of ``rgb_to_yuv420``: (T, 3H/2, W) I420 planes to (T, H, W, 3) uint8 RGB."""
    t, hh, w = planes.shape
    h = hh * 2 // 3
    flat = planes.reshape(t, -1).astype(np.float64)
    n = h * w
    Y = flat[:, :n].reshape(t, h, w)
    U = flat[:, n:n + n // 4].reshape(t, h // 2, w // 2)
    V = flat[:, n + n // 4:].reshape(t, h // 2, w // 2)
    up = lambda a: a.repeat(2, axis=1).repeat(2, axis=2)
    y = (Y - 16.0) / 219.0
    cb, cr = (up(U) - 128.0) / 224.0, (up(V) - 128.0) / 224.0
    r = y + 2.0 * (1.0 - KR) * cr
    b = y + 2.0 * (1.0 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return np.clip(np.round(np.stack([r, g, b], axis=-1) * 255.0), 0, 255).astype(np.uint8)


def frames_sha256(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB over all pixels and channels of two uint8 frame stacks, capped at 100 dB for identical stacks."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else min(100.0, float(10.0 * np.log10(255.0**2 / mse)))


def encode(frames: np.ndarray, path: Path) -> dict:
    """Write ``frames`` ((T, H, W, 3) uint8 RGB) as the registered MP4 and return its record, with the decoded
    frames' hash and their PSNR against ``frames``."""
    import av

    planes = rgb_to_yuv420(frames)
    t, h, w, _ = frames.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(tmp), "w", format="mp4") as container:
        stream = container.add_stream(CODEC, rate=FPS)
        stream.width, stream.height, stream.pix_fmt = w, h, PIX_FMT
        stream.codec_context.thread_count = 1
        stream.codec_context.time_base = Fraction(1, FPS)
        stream.options = {"crf": str(CRF), "preset": PRESET, "x264-params": X264_PARAMS}
        for i in range(t):
            frame = av.VideoFrame.from_ndarray(planes[i], format=PIX_FMT)
            frame.pts = i
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    options = x264_options(tmp)
    if (versions := encoder_versions(tmp)) != VERSIONS:
        raise RuntimeError(f"{tmp}: encoder versions {versions}, registered {VERSIONS}")
    for key, value in REQUIRED_OPTIONS.items():
        if options.get(key) != value:
            raise RuntimeError(f"{tmp}: x264 recorded {key}={options.get(key)}, not the registered {value}")
    tmp.replace(path)
    decoded = decode_rgb(path)
    if decoded.shape != frames.shape:
        raise RuntimeError(f"{path}: decoded {decoded.shape}, encoded {frames.shape}")
    return {"path": str(path), "frames": t, "encoding": ENCODING, "file_bytes": path.stat().st_size,
            "x264_options": {k: options[k] for k in REQUIRED_OPTIONS}, "decoded_frames_sha256": frames_sha256(decoded),
            "psnr_vs_lossless_db": psnr(decoded, frames)}


def encoder_versions(path: Path) -> dict[str, str]:
    """PyAV's and its FFmpeg's versions, and the x264 core version written in the file."""
    import re

    import av

    found = re.search(rb"x264 - core (\d+)", Path(path).read_bytes())
    return {"pyav": av.__version__, "ffmpeg": str(av.ffmpeg_version_info), "x264_core": found.group(1).decode() if found else None}


def x264_options(path: Path) -> dict[str, str]:
    """The settings x264 writes into the stream's SEI ("options: cabac=1 ref=3 ..."), as a dict."""
    data = Path(path).read_bytes()
    start = data.find(b"options: ")
    if start < 0:
        raise RuntimeError(f"{path}: no x264 settings string found")
    end = data.find(b"\x00", start)
    fields = data[start + len(b"options: "):end].decode("ascii", "replace").split()
    return dict(f.split("=", 1) for f in fields if "=" in f)


def decode_yuv420(path: Path) -> np.ndarray:
    """Every frame's I420 planes, (T, 3H/2, W) uint8, exactly as decoded (no colour conversion)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        planes = []
        for frame in container.decode(stream):
            if frame.format.name != PIX_FMT:
                raise RuntimeError(f"{path}: decoded pixel format {frame.format.name}, expected {PIX_FMT}")
            planes.append(frame.to_ndarray(format=PIX_FMT))
    if not planes:
        raise RuntimeError(f"{path}: no frames decoded")
    return np.stack(planes)


def decode_rgb(path: Path) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB frames of a registered MP4, through the numpy BT.709 conversion."""
    return yuv420_to_rgb(decode_yuv420(path))


def encode_lossless(frames: np.ndarray, path: Path) -> dict:
    """Write ``frames`` ((T, H, W, 3) uint8 RGB) as FFV1, then require its decode to equal them bit for bit."""
    import av

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) uint8 frames, got {frames.shape} {frames.dtype}")
    t, h, w, _ = frames.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mkv")
    bgra = np.empty((h, w, 4), np.uint8)
    bgra[..., 3] = 255
    with av.open(str(tmp), "w", format=LOSSLESS["container"]) as container:
        stream = container.add_stream(LOSSLESS["codec"], rate=FPS)
        stream.width, stream.height, stream.pix_fmt = w, h, LOSSLESS["pix_fmt"]
        stream.codec_context.thread_count = LOSSLESS["threads"]
        for i in range(t):
            bgra[..., 0], bgra[..., 1], bgra[..., 2] = frames[i, ..., 2], frames[i, ..., 1], frames[i, ..., 0]
            frame = av.VideoFrame.from_ndarray(bgra, format=LOSSLESS["pix_fmt"])
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    if not np.array_equal(decode_lossless(tmp), frames):
        tmp.unlink()
        raise RuntimeError(f"{path}: the FFV1 file does not decode to the frames it stores")
    tmp.replace(path)
    return {"path": path.name, "frames": t, "format": LOSSLESS, "file_bytes": path.stat().st_size,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "uint8_sha256": frames_sha256(frames), "verified_bitwise": True}


def decode_lossless(path: Path) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB frames of an FFV1 file written by ``encode_lossless``."""
    import av

    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            if frame.format.name != LOSSLESS["pix_fmt"]:
                raise RuntimeError(f"{path}: decoded pixel format {frame.format.name}, expected {LOSSLESS['pix_fmt']}")
            bgra = frame.to_ndarray()
            if not (bgra[..., 3] == 255).all():
                raise RuntimeError(f"{path}: the alpha channel is not the constant 255 encode_lossless writes")
            out.append(bgra[..., [2, 1, 0]])
    if not out:
        raise RuntimeError(f"{path}: no frames decoded")
    return np.ascontiguousarray(np.stack(out))


def load_lossless(path: Path, record: dict) -> np.ndarray:
    """The FFV1 file's frames, checked against the record ``encode_lossless`` returned."""
    frames = decode_lossless(path)
    if frames_sha256(frames) != record["uint8_sha256"]:
        raise RuntimeError(f"{path}: decoded frames differ from the record")
    return frames
