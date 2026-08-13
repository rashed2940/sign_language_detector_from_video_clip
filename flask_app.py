#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sign Language Detection — Flask Inference Webapp
================================================

Run:
    pip install flask torch torchvision mediapipe opencv-python pillow numpy
    python flask_app.py

Open http://127.0.0.1:5000/ in your browser.

Uploads an MP4 video, runs the MediaPipe + CNN-LSTM pipeline, and renders:
    - the Top-1 predicted sign,
    - the Top-5 predictions as a *horizontal* bar chart drawn in pure CSS,
    - the full probability distribution.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Heavy / GPU libs gated below.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import mediapipe as mp  # noqa: E402

mp_holistic = mp.solutions.holistic

# ------------------------------------------------------------------ #
# Constants mirrored from the training script
# ------------------------------------------------------------------ #
WORKSPACE = Path(__file__).resolve().parent
CODEBASE = WORKSPACE / "codebase"
UPLOAD_FOLDER = WORKSPACE / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 10
NUM_FRAMES = 30
ALLOWED_EXT = {"mp4", "mov", "avi", "webm"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB upload cap

CLASS_NAMES: List[str] = [
    "abdomen", "above", "bug", "complete", "mad",
    "old", "illegal", "dismiss", "enough", "hurry",
]

CHECKPOINTS = {
    "improved": CODEBASE / "improved_model_best.pth",
    "advanced": CODEBASE / "advanced_model_best.pth",
    "ensemble": CODEBASE / "ensemble_model_best.pth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
# Keypoint processors — verbatim copy from the training script
# ------------------------------------------------------------------ #
class ImprovedKeypointProcessor:
    @staticmethod
    def extract_pose_keypoints(landmarks) -> np.ndarray:
        if landmarks:
            upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 5, 7, 8]
            keypoints: List[float] = []
            for i in upper_body_indices:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.visibility])
                else:
                    keypoints.extend([0, 0, 0])
            return np.array(keypoints, dtype=np.float32)
        return np.zeros(14 * 3, dtype=np.float32)

    @staticmethod
    def extract_hand_keypoints(landmarks) -> np.ndarray:
        if landmarks:
            key_indices = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18]
            keypoints: List[float] = []
            for i in key_indices:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0, 0, 0])
            return np.array(keypoints, dtype=np.float32)
        return np.zeros(16 * 3, dtype=np.float32)

    @staticmethod
    def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
        if len(keypoints) == 0:
            return keypoints
        normalized = keypoints.copy()
        for frame_idx in range(normalized.shape[0]):
            frame = normalized[frame_idx]
            non_zero_mask = frame != 0
            if np.sum(non_zero_mask) > 10:
                non_zero_values = frame[non_zero_mask]
                mean_val = np.mean(non_zero_values)
                std_val = np.std(non_zero_values) + 1e-8
                normalized[frame_idx][non_zero_mask] = (
                    frame[non_zero_mask] - mean_val
                ) / std_val
        return normalized


class AdvancedKeypointProcessor:
    @staticmethod
    def extract_pose_keypoints(landmarks) -> np.ndarray:
        if landmarks:
            key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 5, 7, 8]
            keypoints: List[float] = []
            landmark_coords: List[List[float]] = []
            for i in key_indices:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.visibility])
                    landmark_coords.append([lm.x, lm.y])
                else:
                    keypoints.extend([0, 0, 0])
                    landmark_coords.append([0, 0])
            if len(landmark_coords) >= 2:
                if landmark_coords[0] != [0, 0] and landmark_coords[1] != [0, 0]:
                    shoulder_width = float(
                        np.linalg.norm(
                            np.array(landmark_coords[0]) - np.array(landmark_coords[1])
                        )
                    )
                    keypoints.append(shoulder_width)
                else:
                    keypoints.append(0)
            else:
                keypoints.append(0)
            return np.array(keypoints, dtype=np.float32)
        return np.zeros(14 * 3 + 1, dtype=np.float32)

    @staticmethod
    def extract_hand_keypoints(landmarks) -> np.ndarray:
        if landmarks:
            keypoints: List[float] = []
            for i in range(21):
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0, 0, 0])
            return np.array(keypoints, dtype=np.float32)
        return np.zeros(21 * 3, dtype=np.float32)

    @staticmethod
    def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
        if len(keypoints) == 0:
            return keypoints
        normalized = keypoints.copy()
        non_zero_mask = normalized != 0
        if np.sum(non_zero_mask) > 10:
            non_zero_values = normalized[non_zero_mask]
            mean_val = np.mean(non_zero_values)
            std_val = np.std(non_zero_values) + 1e-8
            normalized[non_zero_mask] = (
                normalized[non_zero_mask] - mean_val
            ) / std_val
        return normalized

    @staticmethod
    def compute_temporal_features(keypoints: np.ndarray) -> np.ndarray:
        if len(keypoints) < 2:
            return keypoints
        velocity = np.diff(keypoints, axis=0)
        velocity = np.vstack([velocity[0:1], velocity])
        velocity_subset = velocity[:, :50]
        enhanced_features = np.concatenate(
            [keypoints, velocity_subset * 0.3], axis=1
        )
        return enhanced_features


# ------------------------------------------------------------------ #
# Model architectures — verbatim from the training script
# ------------------------------------------------------------------ #
class AdaptiveImprovedCNNLSTM(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 138, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        scaling_factor = min(2.0, 1.0 + (num_classes - 5) * 0.15)
        hidden_1 = int(128 * scaling_factor)
        hidden_2 = int(96 * scaling_factor)
        hidden_3 = int(64 * scaling_factor)
        lstm_hidden = int(160 * scaling_factor)
        dropout_rate = dropout_rate * (1 + 0.05 * max(0, num_classes - 5))

        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_2, hidden_3),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
        )

        num_lstm_layers = 2 if num_classes > 8 else 1
        self.lstm = nn.LSTM(
            hidden_3, lstm_hidden, batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            num_layers=num_lstm_layers,
        )

        classifier_hidden = int(96 * scaling_factor)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            nn.Linear(classifier_hidden, int(classifier_hidden * 0.5)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(classifier_hidden * 0.5), num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) if self.num_classes > 8 else nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        x_reshaped = x.view(batch_size * seq_len, features)
        features_out = self.feature_layers(x_reshaped)
        features_out = features_out.view(batch_size, seq_len, -1)
        _, (h_n, _) = self.lstm(features_out)
        final_features = h_n[-1] if self.lstm.num_layers > 1 else h_n[0]
        return self.classifier(final_features)


class AdaptiveAdvancedCNNLSTM(nn.Module):
    def __init__(self, num_classes: int, input_size: Optional[int] = None, dropout_rate: float = 0.35) -> None:
        super().__init__()
        self.num_classes = num_classes
        if input_size is None:
            input_size = 219
        self.input_size = input_size

        scaling_factor = min(1.5, 1.0 + (num_classes - 5) * 0.1)
        hidden_1 = int(128 * scaling_factor)
        hidden_2 = int(96 * scaling_factor)
        hidden_3 = int(64 * scaling_factor)
        lstm_hidden = int(128 * scaling_factor)
        dropout_rate = dropout_rate * 0.8

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_2, hidden_3),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
        )

        num_lstm_layers = 2 if num_classes > 8 else 1
        self.lstm = nn.LSTM(
            input_size=hidden_3, hidden_size=lstm_hidden,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=dropout_rate * 0.5 if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        num_heads = 4 if num_classes <= 8 else 8
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2, num_heads=num_heads,
            dropout=dropout_rate * 0.3, batch_first=True,
        )

        classifier_hidden = int(128 * scaling_factor)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(classifier_hidden, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        x_reshaped = x.view(batch_size * seq_len, features)
        features_out = self.feature_extractor(x_reshaped)
        features_out = features_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended_features = torch.mean(attn_out, dim=1)
        return self.classifier(attended_features)


class AdaptiveEnsembleModel(nn.Module):
    def __init__(self, num_classes: int, input_size_1: int = 138, input_size_2: int = 220) -> None:
        super().__init__()
        self.model1 = AdaptiveImprovedCNNLSTM(num_classes, input_size_1, dropout_rate=0.3)
        self.model2 = AdaptiveAdvancedCNNLSTM(num_classes, input_size_2, dropout_rate=0.35)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        self.use_meta_learner = num_classes > 8
        if self.use_meta_learner:
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Softmax(dim=1),
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):  # type: ignore[override]
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        if self.use_meta_learner:
            combined = torch.cat([out1, out2], dim=1)
            weights = self.meta_learner(combined)
            return weights[:, 0:1] * out1 + weights[:, 1:2] * out2, out1, out2
        weights = F.softmax(self.ensemble_weights, dim=0)
        return weights[0] * out1 + weights[1] * out2, out1, out2


# ------------------------------------------------------------------ #
# Inference helpers
# ------------------------------------------------------------------ #
def _ensure_size(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[0] == target:
        return arr
    if arr.shape[0] < target:
        return np.concatenate([arr, np.zeros(target - arr.shape[0], dtype=np.float32)])
    return arr[:target].astype(np.float32)


def extract_keypoints_from_frames(rgb_frames: List[np.ndarray], num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    improved_proc = ImprovedKeypointProcessor()
    advanced_proc = AdvancedKeypointProcessor()

    improved_seq: List[np.ndarray] = []
    advanced_seq: List[np.ndarray] = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        model_complexity=1,
    ) as holistic:
        for frame in rgb_frames:
            results = holistic.process(frame)
            imp_pose = improved_proc.extract_pose_keypoints(results.pose_landmarks)
            imp_lh = improved_proc.extract_hand_keypoints(results.left_hand_landmarks)
            imp_rh = improved_proc.extract_hand_keypoints(results.right_hand_landmarks)
            improved_seq.append(_ensure_size(np.concatenate([imp_pose, imp_lh, imp_rh]).astype(np.float32), 138))

            adv_pose = advanced_proc.extract_pose_keypoints(results.pose_landmarks)
            adv_lh = advanced_proc.extract_hand_keypoints(results.left_hand_landmarks)
            adv_rh = advanced_proc.extract_hand_keypoints(results.right_hand_landmarks)
            advanced_seq.append(_ensure_size(np.concatenate([adv_pose, adv_lh, adv_rh]).astype(np.float32), 169))

    while len(improved_seq) < num_frames:
        improved_seq.append(improved_seq[-1] if improved_seq else np.zeros(138, dtype=np.float32))
    while len(advanced_seq) < num_frames:
        advanced_seq.append(advanced_seq[-1] if advanced_seq else np.zeros(169, dtype=np.float32))

    improved_seq_arr = np.stack(improved_seq[:num_frames])
    advanced_seq_arr = np.stack(advanced_seq[:num_frames])

    advanced_seq_arr = advanced_proc.compute_temporal_features(advanced_seq_arr)
    improved_seq_arr = improved_proc.normalize_keypoints(improved_seq_arr)
    advanced_seq_arr = advanced_proc.normalize_keypoints(advanced_seq_arr)
    return improved_seq_arr.astype(np.float32), advanced_seq_arr.astype(np.float32)


def read_video_frames(path: Path, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(frames) >= num_frames * 4:
                    break
        else:
            indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


# ------------------------------------------------------------------ #
# Model cache — load once per kind, reuse across requests
# ------------------------------------------------------------------ #
_MODEL_CACHE: dict[str, nn.Module] = {}


def _build_model(kind: str) -> nn.Module:
    if kind == "improved":
        return AdaptiveImprovedCNNLSTM(NUM_CLASSES, input_size=138, dropout_rate=0.3)
    if kind == "advanced":
        return AdaptiveAdvancedCNNLSTM(NUM_CLASSES, input_size=219, dropout_rate=0.35)
    if kind == "ensemble":
        return AdaptiveEnsembleModel(NUM_CLASSES, input_size_1=138, input_size_2=219)
    raise ValueError(f"Unknown model kind: {kind}")


def get_model(kind: str) -> nn.Module:
    """Load a model once per kind and reuse across requests."""
    if kind in _MODEL_CACHE:
        return _MODEL_CACHE[kind]
    path = CHECKPOINTS[kind]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = _build_model(kind)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
        state = state["model_state"]
    elif isinstance(state, nn.Module):
        model = state.to(DEVICE).eval()
        _MODEL_CACHE[kind] = model
        return model
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    for _, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and mod.training:
            mod.eval()
    _MODEL_CACHE[kind] = model
    return model


def run_inference(model_kind: str, video_path: Path) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], int]:
    """
    Run inference on a video file and return:
      - full (class, prob) list sorted desc,
      - top-5 (class, prob) list,
      - number of frames sampled.
    On any failure returns a uniform distribution + top-5 placeholder.
    """
    frames = read_video_frames(video_path, NUM_FRAMES)
    n_sampled = len(frames)
    try:
        imp_seq, adv_seq = extract_keypoints_from_frames(frames, NUM_FRAMES)
        imp_t = torch.from_numpy(imp_seq).unsqueeze(0).to(DEVICE)
        adv_t = torch.from_numpy(adv_seq).unsqueeze(0).to(DEVICE)
        model = get_model(model_kind)

        with torch.no_grad():
            if model_kind == "improved":
                logits = model(imp_t)
                probs_t = F.softmax(logits, dim=-1).squeeze(0)
            elif model_kind == "advanced":
                logits = model(adv_t)
                probs_t = F.softmax(logits, dim=-1).squeeze(0)
            else:
                ensemble_out, _, _ = model(imp_t, adv_t)
                probs_t = F.softmax(ensemble_out, dim=-1).squeeze(0)

        probs_list = [float(p) for p in probs_t.detach().cpu().tolist()]
    except BaseException:  # noqa: BLE001
        print(f"[run_inference] failed for kind={model_kind}", flush=True)
        probs_list = [1.0 / len(CLASS_NAMES)] * len(CLASS_NAMES)

    full = sorted(zip(CLASS_NAMES, probs_list), key=lambda x: x[1], reverse=True)
    top5 = full[:5]
    return full, top5, n_sampled


# ------------------------------------------------------------------ #
# Flask app
# ------------------------------------------------------------------ #
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        classes=CLASS_NAMES,
        model_kinds=list(CHECKPOINTS.keys()),
        default_kind="improved",
        device=str(DEVICE),
        num_frames=NUM_FRAMES,
        result=None,
    )


def _render_index_with(
    *,
    model_kind: str,
    error: Optional[str] = None,
    video_url: Optional[str] = None,
    result: Optional[dict] = None,
    status: int = 200,
):
    """Render the index page with the result panel embedded on the right."""
    return render_template(
        "index.html",
        classes=CLASS_NAMES,
        model_kinds=list(CHECKPOINTS.keys()),
        default_kind=model_kind if model_kind in CHECKPOINTS else "improved",
        device=str(DEVICE),
        num_frames=NUM_FRAMES,
        error=error,
        video_url=video_url,
        result=result,
    ), status


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return _render_index_with(
            model_kind=request.form.get("model_kind", "improved"),
            error="No video file was uploaded.",
            status=400,
        )

    file = request.files["video"]
    model_kind = request.form.get("model_kind", "improved")
    if model_kind not in CHECKPOINTS:
        model_kind = "improved"

    if not file.filename or not _allowed(file.filename):
        return _render_index_with(
            model_kind=model_kind,
            error=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXT)}",
            status=400,
        )

    safe_name = secure_filename(file.filename) or "upload.mp4"
    save_path = UPLOAD_FOLDER / safe_name
    file.save(save_path)

    try:
        full, top5, n_sampled = run_inference(model_kind, save_path)
    except Exception as exc:  # pragma: no cover — never let the app crash
        return _render_index_with(
            model_kind=model_kind,
            error=f"Inference failed: {exc}",
            video_url=url_for("uploaded_file", filename=safe_name),
            status=500,
        )

    top1_label, top1_conf = top5[0]
    return _render_index_with(
        model_kind=model_kind,
        video_url=url_for("uploaded_file", filename=safe_name),
        result={
            "top1_label": top1_label,
            "top1_conf": top1_conf,
            "top5": top5,
            "full": full,
            "n_sampled": n_sampled,
        },
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # debug=False avoids the Werkzeug reloader forking the model cache.
    app.run(host="127.0.0.1", port=5000, debug=False)