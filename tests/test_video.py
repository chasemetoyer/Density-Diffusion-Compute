"""The registered video encoding: CRF 10 x264 with the colour conversion in numpy, decoded frames compared by hash."""

import av
import numpy as np
import pytest

from density import video


def clip(t=9, h=64, w=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 2, yy * 3, (xx + yy)], axis=-1)
    return np.stack([(base + 4 * i + rng.integers(0, 3, (h, w, 3))) % 256 for i in range(t)]).astype(np.uint8)


def test_round_trip_records_settings_hash_and_psnr(tmp_path):
    frames = clip()
    record = video.encode(frames, tmp_path / "a.mp4")
    decoded = video.decode_rgb(tmp_path / "a.mp4")
    assert decoded.shape == frames.shape and decoded.dtype == np.uint8
    assert record["frames"] == 9 and record["x264_options"] == {"crf": "10.0", "threads": "1"}
    assert record["decoded_frames_sha256"] == video.frames_sha256(decoded)
    assert record["psnr_vs_lossless_db"] == video.psnr(decoded, frames) > 30
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_encoding_twice_gives_identical_decoded_frames(tmp_path):
    frames = clip()
    a = video.encode(frames, tmp_path / "a.mp4")
    b = video.encode(frames, tmp_path / "b.mp4")
    assert a["decoded_frames_sha256"] == b["decoded_frames_sha256"]
    assert video.frames_sha256(video.decode_rgb(tmp_path / "a.mp4")) == a["decoded_frames_sha256"]


def test_stream_is_tagged_bt709_limited_range_at_16_fps(tmp_path):
    video.encode(clip(), tmp_path / "a.mp4")
    with av.open(str(tmp_path / "a.mp4")) as container:
        stream = container.streams.video[0]
        cc = stream.codec_context
        assert (cc.color_primaries, cc.color_trc, cc.colorspace, cc.color_range) == (1, 1, 1, 1)  # BT.709, MPEG range
        assert stream.average_rate == video.FPS and stream.codec_context.pix_fmt == video.PIX_FMT


def test_colour_conversion_inverts_within_quantization():
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, (2, 32, 48, 3), dtype=np.uint8).repeat(2, axis=1).repeat(2, axis=2)  # chroma subsampling loses nothing
    back = video.yuv420_to_rgb(video.rgb_to_yuv420(blocks))
    assert np.abs(back.astype(int) - blocks).max() <= 2
    grey = np.full((1, 2, 2, 3), 128, dtype=np.uint8)
    planes = video.rgb_to_yuv420(grey)
    assert planes.shape == (1, 3, 2) and planes[0, 2:].tolist() == [[128, 128]]  # neutral chroma
    assert video.rgb_to_yuv420(np.zeros((1, 2, 2, 3), np.uint8))[0, 0, 0] == 16 and video.rgb_to_yuv420(np.full((1, 2, 2, 3), 255, np.uint8))[0, 0, 0] == 235


def test_chroma_is_the_mean_of_each_2x2_block():
    frames = np.zeros((1, 2, 2, 3), np.uint8)
    frames[0, 0, 0] = (255, 0, 0)  # one red pixel in a black block
    cr = video.rgb_to_yuv420(frames)[0, 2, 1]
    assert cr == round(128 + 224 * (1 - video.KR) / (2 * (1 - video.KR)) / 4)


def test_psnr():
    a = np.zeros((1, 4, 4, 3), np.uint8)
    assert video.psnr(a, a) == 100.0
    b = a.copy()
    b[...] = 1  # MSE 1
    assert video.psnr(a, b) == pytest.approx(10 * np.log10(255.0**2))


@pytest.mark.parametrize("shape, dtype", [((2, 5, 4, 3), np.uint8), ((2, 4, 4, 3), np.float32), ((2, 4, 4, 4), np.uint8)])
def test_refuses_odd_sizes_and_non_uint8(shape, dtype, tmp_path):
    with pytest.raises(ValueError):
        video.encode(np.zeros(shape, dtype), tmp_path / "a.mp4")


def test_lossless_copy_decodes_bit_for_bit(tmp_path):
    frames = clip(t=5, seed=3)
    record = video.encode_lossless(frames, tmp_path / "a.mkv")
    assert record["verified_bitwise"] and record["uint8_sha256"] == video.frames_sha256(frames)
    assert np.array_equal(video.load_lossless(tmp_path / "a.mkv", record), frames)
    assert not list(tmp_path.glob("*.tmp.mkv"))
    with pytest.raises(RuntimeError, match="differ from the record"):
        video.load_lossless(tmp_path / "a.mkv", {**record, "uint8_sha256": "0" * 64})
    with pytest.raises(ValueError):
        video.encode_lossless(frames.astype(np.float32), tmp_path / "b.mkv")
