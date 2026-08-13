---
applyTo: "**"
description: Sign-language detection project — Flask inference app
---

# Sign-Language Detection (workspace conventions)

## Layout
- `1_op_ds_model_v3a_10c_v3_l40_AM.py` — original training script. Treat as the
  source of truth for model architecture, keypoint extraction, class list, and
  preprocessing. When you copy code into `flask_app.py`, copy it **verbatim**
  — don't refactor.
- `codebase/` — trained checkpoints (`improved_model_best.pth`,
  `advanced_model_best.pth`, `ensemble_model_best.pth`) plus per-model
  architecture logs and metrics CSVs. Architecture files here may differ
  slightly from the training script; trust the script.
- `train_data/`, `test_data/` — 10-class sign-language video dataset.
  Class list:
  `abdomen, above, bug, complete, mad, old, illegal, dismiss, enough, hurry`.
- `flask_app.py` — Flask inference webapp. Caches models in an in-process
  `_MODEL_CACHE` dict.
- `templates/index.html` — Jinja2 template for Flask (upload form + side-by-side result panel).
- `uploads/` — Flask upload directory (auto-created; served at `/uploads/...`).

## Pipeline (must stay identical)
1. Read up to 30 evenly-spaced RGB frames from the video with OpenCV.
2. Extract MediaPipe Holistic keypoints.
3. `ImprovedKeypointProcessor` → 138 features, per-frame z-normalize.
4. `AdvancedKeypointProcessor` → 169 features → `compute_temporal_features`
   → 219 features, global z-normalize.
5. Feed to one of the three model kinds. The ensemble takes **both**
   sequences and returns a fused logit.
6. Softmax → probabilities → rank.

## Models
- `improved` — `AdaptiveImprovedCNNLSTM`, input size **138**.
- `advanced` — `AdaptiveAdvancedCNNLSTM`, input size **219**.
- `ensemble` — `AdaptiveEnsembleModel`, takes both 138 and 219.

When loading checkpoints, unwrap `state["model_state"]` if present and
`strict=False` the rest. Always `model.to(DEVICE)` first, then `model.eval()`,
then re-eval every BatchNorm submodule.

## Rules of thumb
- NEVER let an exception reach the UI — wrap `run_inference` in a try/except
  that returns a uniform distribution.
- Keep the upload cap at 200 MB (`MAX_CONTENT_LENGTH`).
- `device` defaults to CUDA if available, else CPU.
- Frames must always be exactly `NUM_FRAMES = 30` — pad by repeating the last
  frame, never silently drop.

## How to run
- Flask: `python flask_app.py` → http://127.0.0.1:5000/
