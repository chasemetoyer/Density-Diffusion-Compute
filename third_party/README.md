# Vendored code

## hands23_detector

Hands23 hand, contact and object detector (Cheng et al., NeurIPS 2023), MIT license (`hands23_detector/LICENSE`).

- Source: https://github.com/EvaCheng-cty/hands23_detector at commit `1bc0f919ffa7a9f375e7e8042e1d26f3743d5819`.
- Left out of the copy: `demo_example/` (sample images and results), `.gitignore`, compiled `.pyc` files, and
  `utils/times_b.ttf`, a Hewlett-Packard Times Bold font ("All rights reserved") that the MIT license does not
  cover. Only `utils/vis_utils.vis_per_image` opens that font, and this project never calls it.
- `patches/hands23_detector.diff` records every change to the upstream files: `.cuda()` calls become
  `.to(device)`, `pdb.set_trace()` calls become errors, and Pillow's removed `textsize` becomes `textbbox`.

To reproduce the copy:

```bash
git clone https://github.com/EvaCheng-cty/hands23_detector.git && cd hands23_detector
```

```bash
git checkout 1bc0f919ffa7a9f375e7e8042e1d26f3743d5819 && rm -rf .git .gitignore demo_example utils/times_b.ttf
```

```bash
git apply ../third_party/patches/hands23_detector.diff
```

The weights are not vendored: `src/density/masks/weights.py` records the original file and the converted
safetensors copy that the pipeline loads.

## Milestone 5: the five VBench quality dimensions

Protocol v2 section 7 scores subject consistency, background consistency, motion smoothness, aesthetic
quality and imaging quality with VBench's own code. VBench 0.1.5 cannot be installed into this environment:
- it pins numpy below 2 and transformers 4.33.2;
- its AMT loader fails under torch 2.14's data-only default;
- decord has no macOS arm64 wheel;
- its source build rejects CUDA 13.

So the code each dimension runs is vendored here, verbatim at pinned tags, except where a diff in `patches/`
records the change. `src/density/metrics/quality.py` builds every model from verified, data-only weight files
(`src/density/metrics/weights.py`) and calls VBench's per-video functions.

Two of these components are **non-commercial**: AMT (CC BY-NC 4.0) and IQA-PyTorch (CC BY-NC-SA 4.0). They
are used here for research only.

| Folder | Source, pinned | License | Tarball SHA-256 |
| --- | --- | --- | --- |
| `vbench/` | https://github.com/Vchitect/VBench, tag v0.1.5 = commit `1039fe66d7aa2f4b214003ca2d35f5011c48bfee` | Apache-2.0 (`vbench/LICENSE`); AMT CC BY-NC 4.0 (`vbench/vbench/third_party/amt/LICENSE`) | `f64d13fa047063e0eb8d730124ce2e8cce8c407c14f9b216a212df801b47104c` |
| `clip/` | https://github.com/openai/CLIP, commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | MIT | `553c80a6364ac7350dd7800f535531175f22c11fa742bda1b50b62121eb5f40f` |
| `dino/` | https://github.com/facebookresearch/dino, commit `7c446df5b9f45747937fb0d72314eb9f7b66930a` | Apache-2.0 | `b4e56a51a36a10b5d0a26ce44966efd4499b7cf08ce8675a04127780d05394d1` |
| `pyiqa/` | https://github.com/chaofengc/IQA-PyTorch, tag v0.1.10 = commit `3041923434fa170eac0ba4182e6584dcb7a9fefc` | CC BY-NC-SA 4.0 | `336d9fc35a3fe2fb69362636892d9a06300e5d48dd6bdcfef735a19a2261fc52` |

What each folder holds, and what each diff changes:
- **`vbench/`**
  - **Holds:** `LICENSE`; `vbench/__init__.py`, `utils.py`, `distributed.py`, and the five dimension modules; and
    the AMT-S inference subset of `vbench/third_party/amt`:
    - `LICENSE`, `cfgs/AMT-S.yaml`;
    - `networks/__init__.py`, `networks/AMT-S.py`, `networks/blocks/*.py`;
    - `utils/{__init__,build_utils,dist_utils,flow_utils,utils}.py`.
  - **Left out:** every other dimension, the CLI and prompts, ViCLIP, RAFT, GRiT, tag2Text and UMT, and AMT's
    training code and other configs.
  - **`patches/vbench.diff`:**
    - `load_video` reads MP4s with the registered decoder (`density.video.decode_rgb`) instead of decord.
    - Motion smoothness takes its frames from the same decoder instead of OpenCV.
    - AMT-S is imported by its full module name, rather than through the `sys.path` entry `build_utils`
      appends, and is loaded from the converted safetensors file.
- **`clip/`**
  - **Holds:** `LICENSE` and `clip/model.py`, which provides `build_model`.
  - **`patches/clip.diff`:** replaces `clip/__init__.py`. The downloader, TorchScript loader and tokenizer are
    not vendored; the tokenizer needs ftfy, which this project must not install.
- **`dino/`**
  - **Holds:** `LICENSE`, plus `vision_transformer.py` and `utils.py`, moved from the repository root into
    `dino_vit/` (new `__init__.py`).
  - **`patches/dino.diff`:** makes the `utils` import package-relative, so DINO's `utils` can't collide with
    another top-level `utils` module (Hands23 has one). `hubconf.py` is not vendored; its `dino_vitb16` steps
    (`vit_base(patch_size=16, num_classes=0)`, then a strict state-dict load) are in
    `density/metrics/quality.py`.
- **`pyiqa/`**
  - **Holds:** `LICENSE` and MUSIQ's modules: `pyiqa/archs/{musiq_arch,arch_util,constants}.py`,
    `pyiqa/data/multiscale_trans_util.py`, `pyiqa/matlab_utils/*`, and
    `pyiqa/utils/{registry,download_util,misc,dist_util}.py`.
  - **`patches/pyiqa.diff`:** replaces the four package `__init__` files, which import every architecture,
    dataset, logger and model. The checkpoint loads with torch 2.14's data-only default.

To reproduce a folder, take the upstream tarball at its pinned commit, keep the files listed above (moving
DINO's two files into `dino_vit/`), and apply that folder's diff from inside it:

```bash
curl -fsSL -o vbench.tar.gz https://codeload.github.com/Vchitect/VBench/tar.gz/1039fe66d7aa2f4b214003ca2d35f5011c48bfee
```

```bash
git apply -p1 --directory=third_party/vbench third_party/patches/vbench.diff
```
