# Foveated 2DGS — Agent Guide

Incremental research fork of [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting).  
Goal: integrate foveated (gaze-aware) perception into 2DGS training and rendering.  
Supervised by Kaan Akşit and Rafał Mantiuk.

---

## Review-before-apply workflow

**Before making any code change, Claude must:**

1. Show a **feature-level summary** — what the change does and why
2. Show **per-file details** — each file, what lines change, and what each change does
3. **Wait for confirmation** before applying anything

If a `TODO.md` or `Plan.md` exists in this repo, read it first to check current progress before proposing any changes.

---

## Planned work tracks

### Track 1 — Foveated loss in training `[ ]`
Add `MetamerMSELoss` from [odak](https://github.com/kaanaksit/odak) as an optional weighted loss term in `train.py`, alongside the existing L1 + SSIM loss.

Files to change:
- `arguments/__init__.py` — add `lambda_fovea`, `gaze_x`, `gaze_y`, `fovea_start_iter`
- `train.py` — import and initialise `MetamerMSELoss`, add foveal loss into training loop

Usage once implemented:
```bash
python train.py -s <scene> -m <output> \
  --lambda_fovea 0.1 --gaze_x 0.5 --gaze_y 0.5 --fovea_start_iter 3000
```
Default `lambda_fovea=0.0` keeps behaviour identical to base repo.

### Track 2 — FovVideoVDP evaluation `[ ]`
Use [FovVideoVDP](https://github.com/gfxdisp/FovVideoVDP) as a perceptual quality metric to evaluate rendered images against GT, complementing the existing PSNR/SSIM/LPIPS/MetamerLoss metrics.

### Track 3 — Rendering pipeline optimisation `[ ]`
Reduce peripheral Gaussian count for faster foveated rendering. Candidate approaches:
- Eccentricity-based Gaussian culling at render time
- Variable SH degree by eccentricity zone
- LOD merging of peripheral Gaussians

---

## Key files

| File | Purpose |
|------|---------|
| `train.py` | Main training loop — loss computation at lines 73–88 |
| `arguments/__init__.py` | Training hyperparameters — lambda values at lines 85–87 |
| `render.py` | Renders trained model to images |
| `gaussian_renderer/` | CUDA rasterization |
| `scene/` | Scene and camera loading |
| `utils/loss_utils.py` | L1 and SSIM loss functions |

---

## Running baseline training (no foveated loss)

```bash
conda activate surfel_splatting
cd /home/aidcheng/Developer/2dgs/foveated-2dgs

python train.py -s <path/to/DTU/scanXXX> -m output/scanXXX -r 2 --depth_ratio 1
python render.py -s <path/to/DTU/scanXXX> -m output/scanXXX -r 2 --depth_ratio 1 --skip_mesh
```

---

## Environment

- Conda env: `surfel_splatting` (Python 3.8, PyTorch 2.1.0+cu121)
- GPU: RTX 4070 Ti
- CUDA: 12.1
