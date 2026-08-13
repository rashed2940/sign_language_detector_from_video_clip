# 🤟 Sign Language Detection

A deep learning-based sign language recognition system that detects and classifies hand signs from video input using MediaPipe keypoints and CNN-LSTM neural networks.

**Author:** [Sk Rashed Bin Mohammad](https://github.com/rashed2940) · rashed2940@gmail.com

## Overview

This project implements an end-to-end sign language detection pipeline:
- **Video Processing**: Extracts frames from MP4/MOV/AVI/WebM videos
- **Keypoint Extraction**: Uses MediaPipe Holistic to extract pose and hand keypoints
- **Feature Engineering**: Applies advanced keypoint processing with temporal features
- **Deep Learning Models**: Trains and deploys three model architectures:
  - **Improved Model**: CNN-LSTM with 138-dimensional features
  - **Advanced Model**: CNN-LSTM with 219-dimensional features + temporal analysis
  - **Ensemble Model**: Combines both models for enhanced accuracy
- **Flask Web Interface**: Upload videos and get real-time predictions with confidence scores

## Features

✅ Multi-model architecture (Improved, Advanced, Ensemble)  
✅ Flask web app with upload functionality  
✅ Real-time video inference  
✅ Top-5 prediction visualization  
✅ Full probability distribution display  
✅ GPU acceleration support (CUDA)  
✅ Robust error handling  
✅ 200 MB upload size limit  

## Supported Sign Classes

The model recognizes **10 sign language classes**:

```
1. Abdomen     6. Old
2. Above       7. Illegal
3. Bug         8. Dismiss
4. Complete    9. Enough
5. Mad         10. Hurry
```

## Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (optional, for GPU acceleration)
- **PyTorch** with CUDA support (or CPU version)
- **FFmpeg** (for video processing)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Sign_language_detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Or manually install the key packages:
```bash
pip install flask torch torchvision mediapipe opencv-python pillow numpy
```

### 4. Verify installation
```bash
python -c "import torch; print(f'PyTorch installed. GPU available: {torch.cuda.is_available()}')"
```

## Project Structure

```
Sign_language_detection/
├── README.md                                  # This file
├── requirements.txt                           # Project dependencies
├── flask_app.py                               # Flask inference webapp
├── 1_op_ds_model_v3a_10c_v3_l40_AM.py        # Training script (source of truth)
├── templates/
│   └── index.html                            # Web UI template
├── uploads/                                  # User-uploaded videos (auto-created)
├── codebase/
│   ├── improved_model_best.pth               # Improved model checkpoint
│   ├── advanced_model_best.pth               # Advanced model checkpoint
│   ├── ensemble_model_best.pth               # Ensemble model checkpoint
│   ├── 1_op_ds_model_v3a_10c_v3_l40_AM.py   # Architecture reference
│   ├── improved_cnnlstm_architecture.txt     # Model architecture docs
│   ├── advanced_cnnlstm_architecture.txt
│   ├── ensemble_model_architecture.txt
│   ├── improved_cnnlstm_detailed_metrics.csv # Training metrics
│   ├── advanced_cnnlstm_detailed_metrics.csv
│   ├── ensemble_detailed_metrics.csv
│   └── model_comparison.csv                  # Performance comparison
├── train_data/                               # Training dataset (10 classes)
│   ├── abdomen/ ├── above/ ├── bug/
│   ├── complete/ ├── mad/ ├── old/
│   ├── illegal/ ├── dismiss/ ├── enough/
│   └── hurry/
├── test_data/                                # Test dataset (10 classes)
│   └── [same structure as train_data]
└── .gitignore                                # Git ignore patterns
```

## Dataset

This project uses a curated 10-class subset derived from the **WLASL (Word-Level American Sign Language)** dataset, the largest publicly available video dataset for word-level ASL recognition.

### Source Dataset

| Property | Value |
|----------|-------|
| **Name** | WLASL — Word-Level American Sign Language |
| **Source** | [Kaggle: Sign Language Dataset (WLASL Videos)](https://www.kaggle.com/datasets/waseemnagahhenes/sign-language-dataset-wlasl-videos) |
| **Original Size** | ~12,000 videos across 2,000 ASL words |
| **Full Dataset** | [Official WLASL repository](https://github.com/dxli94/WLASL) |
| **Use Case** | Word-level action recognition, sign language translation research |
| **License** | As specified by WLASL authors (Dongxu Li & Hongdong Li) |

### Our 10-Class Subset

From the 2,000-word WLASL corpus, this project selects **10 word classes** for focused experimentation:

| # | Class | Description |
|---|-------|-------------|
| 1 | abdomen | Sign for "abdomen" |
| 2 | above | Sign for "above" |
| 3 | bug | Sign for "bug" |
| 4 | complete | Sign for "complete" |
| 5 | mad | Sign for "mad" |
| 6 | old | Sign for "old" |
| 7 | illegal | Sign for "illegal" |
| 8 | dismiss | Sign for "dismiss" |
| 9 | enough | Sign for "enough" |
| 10 | hurry | Sign for "hurry" |

### Acknowledgments

> **WLASL** is the largest video dataset for Word-Level American Sign Language (ASL) recognition, featuring 2,000 common different words in ASL. We hope WLASL will facilitate research in sign language understanding and eventually benefit the communication between deaf and hearing communities.
>
> **Original Authors:** Dongxu Li and Hongdong Li
> **Please cite the WLASL paper and visit the official website and repository when using this dataset.**

```bibtex
@inproceedings{li2020wlasl,
  title={Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  year={2020}
}
```

## Inference Pipeline

The sign detection pipeline follows this exact sequence:

1. **Frame Extraction**
   - Read up to 30 evenly-spaced RGB frames from the video
   - Pad with the last frame if video has fewer than 30 frames

2. **Keypoint Extraction**
   - Use MediaPipe Holistic to extract:
     - Pose keypoints (upper body focus)
     - Hand keypoints (left & right hands)

3. **Feature Processing**
   - **ImprovedKeypointProcessor**: 138 features, per-frame z-normalization
   - **AdvancedKeypointProcessor**: 169 features → temporal features → 219 features, global z-normalization

4. **Model Inference**
   - Feed feature sequences to selected model:
     - `improved`: Uses 138-dim features
     - `advanced`: Uses 219-dim features
     - `ensemble`: Uses both 138-dim and 219-dim, fuses logits
   - Apply softmax to get probability distribution

5. **Output**
   - Top-1 prediction
   - Top-5 predictions with confidence bars
   - Full class probability distribution

## Quick Start

### Run the Flask App

```bash
python flask_app.py
```

The app starts at **http://127.0.0.1:5000/**

### Usage Steps

1. **Open the web interface** in your browser
2. **Select a model**: Choose between "Improved", "Advanced", or "Ensemble"
3. **Upload a video**: MP4, MOV, AVI, or WebM (max 200 MB)
4. **View results**: See top predictions, confidence scores, and probability charts

### Example: Programmatic Inference

```python
from flask_app import run_inference
import torch

# Run inference with the ensemble model
predictions, class_names = run_inference(
    video_path='path/to/video.mp4',
    model_type='ensemble',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# predictions: numpy array of shape (10,) - probability distribution
# class_names: list of class names
print(f"Top prediction: {class_names[predictions.argmax()]}")
print(f"Confidence: {predictions.max():.2%}")
```

## Model Architecture

### Improved Model (AdaptiveImprovedCNNLSTM)
- **Input size**: 138 features per frame
- **Architecture**: 1D CNN → LSTM → Fully Connected layers
- **Best for**: Fast inference, reasonable accuracy

### Advanced Model (AdaptiveAdvancedCNNLSTM)
- **Input size**: 219 features (poses + hands + temporal features)
- **Architecture**: 1D CNN → LSTM → Fully Connected layers
- **Temporal features**: Velocity, acceleration, and time-series patterns
- **Best for**: High accuracy with temporal analysis

### Ensemble Model (AdaptiveEnsembleModel)
- **Input**: Both 138-dim and 219-dim sequences
- **Architecture**: Dual-path processing with feature fusion
- **Output**: Fused logits combining both models
- **Best for**: Maximum accuracy, combines strengths of both models

## Model Loading and Checkpoint Handling

The app implements robust checkpoint loading:

```python
# Checkpoints are stored in state["model_state"] if they were saved with wrapper
# Always load with strict=False to handle architecture variations
checkpoint = torch.load(model_path, map_location=DEVICE)
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint
    
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE).eval()

# Re-evaluate BatchNorm layers
for module in model.modules():
    if isinstance(module, nn.BatchNorm1d):
        module.eval()
```

## Configuration

### Environment Variables

```bash
# Device selection (auto-detects GPU)
CUDA_VISIBLE_DEVICES=0

# Flask settings
FLASK_ENV=production
FLASK_DEBUG=0

# Upload limits
MAX_CONTENT_LENGTH=200000000  # 200 MB in bytes
```

### Key Constants (in flask_app.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `NUM_FRAMES` | 30 | Frames to extract from video |
| `NUM_CLASSES` | 10 | Number of sign classes |
| `MAX_CONTENT_LENGTH` | 200 MB | Max upload size |
| `ALLOWED_EXT` | mp4, mov, avi, webm | Supported video formats |

## Training

To retrain models on your own dataset:

```bash
python 1_op_ds_model_v3a_10c_v3_l40_AM.py train \
    "path/to/dataset" \
    --model_type all \
    --epochs 50 \
    --batch_size 8 \
    --num_frames 30 \
    --num_classes 10 \
    --output_dir "codebase"
```

**Important**: The training script is the **source of truth** for model architecture, keypoint extraction, and preprocessing. Always copy code from the training script verbatim to ensure pipeline consistency.

## Performance Metrics

Check the metrics CSVs in `codebase/`:
- `improved_cnnlstm_detailed_metrics.csv` - Improved model metrics
- `advanced_cnnlstm_detailed_metrics.csv` - Advanced model metrics  
- `ensemble_detailed_metrics.csv` - Ensemble model metrics
- `model_comparison.csv` - Side-by-side comparison

## Troubleshooting

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

### Model Loading Errors
- Ensure checkpoint files exist in `codebase/` directory
- Check DEVICE is set correctly (CUDA/CPU)
- Verify model architecture matches checkpoint

### Video Processing Issues
- **No frames extracted**: Check video file format and codec
- **MediaPipe errors**: Ensure video is valid and not corrupted
- **Keypoint extraction fails**: Video may have no visible hands/pose

### Flask App Won't Start
```bash
# Check port 5000 is available
netstat -ano | findstr :5000  # Windows
lsof -i :5000  # macOS/Linux

# Run on different port
python flask_app.py --port 8080
```

## Error Handling

The app implements comprehensive error handling:
- **Malformed videos**: Returns uniform distribution (0.1 probability per class)
- **Missing models**: Skips to fallback model
- **GPU memory errors**: Automatically falls back to CPU
- **Upload errors**: Clear user-facing error messages

## Technical Details

### Keypoint Normalization
- **Per-frame z-normalization** (Improved model): Normalizes each frame independently
- **Global z-normalization** (Advanced model): Normalizes across entire sequence

### Temporal Features
The Advanced model computes additional temporal features:
- Velocity (frame-to-frame change)
- Acceleration (velocity change)
- Time-series patterns using windowed statistics

### Batch Normalization
All BatchNorm layers are re-evaluated in inference mode to ensure consistent predictions:
```python
for module in model.modules():
    if isinstance(module, nn.BatchNorm1d):
        module.eval()
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=1.10.0 | Deep learning framework |
| torchvision | >=0.11.0 | Computer vision utilities |
| flask | >=2.0.0 | Web framework |
| mediapipe | >=0.8.0 | Keypoint extraction |
| opencv-python | >=4.5.0 | Video processing |
| numpy | >=1.19.0 | Numerical computing |
| pillow | >=8.0.0 | Image processing |

See `requirements.txt` for complete list.

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{sign_language_detection_2024,
  title={Sign Language Detection: Deep Learning-based Recognition System},
  author={Sk Rashed Bin Mohammad},
  year={2024},
  url={https://github.com/rashed2940/sign_language_detector_from_video_clip}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues, questions, or suggestions:
- Open an GitHub issue
- Check existing issues for similar problems
- Provide error messages and system details

## Contact

- **Author:** Sk Rashed Bin Mohammad
- **GitHub:** [@rashed2940](https://github.com/rashed2940)
- **Email:** rashed2940@gmail.com

---

**Last Updated**: August 2024  
**Version**: 1.0.0  
**Author:** Sk Rashed Bin Mohammad
