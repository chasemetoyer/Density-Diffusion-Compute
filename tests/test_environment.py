"""Job metadata records the GPU model and driver version on CUDA machines."""

import subprocess

from density.checks import euler


def test_gpu_info_is_none_without_cuda(monkeypatch):
    monkeypatch.setattr(euler.torch.cuda, "is_available", lambda: False)
    assert euler.gpu_info() is None
    assert euler.environment()["gpu"] is None


def test_gpu_info_reads_model_and_driver(monkeypatch):
    monkeypatch.setattr(euler.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(euler.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(euler.torch.cuda, "get_device_name", lambda i: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(euler.torch.backends.cudnn, "version", lambda: 90100)

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout="NVIDIA GeForce RTX 4090, 570.124.06\n", stderr="")

    monkeypatch.setattr(euler.subprocess, "run", fake_run)
    info = euler.gpu_info()
    assert info["devices"] == ["NVIDIA GeForce RTX 4090"]
    assert info["nvidia_smi"] == [{"name": "NVIDIA GeForce RTX 4090", "driver_version": "570.124.06"}]
    assert info["cudnn"] == 90100
