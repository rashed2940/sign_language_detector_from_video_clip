#!/usr/bin/env python3
# Enhanced Multi-Model Sign Language Recognition with Adaptive Class Handling
# Fixed Advanced model issues
# Usage:
# python script.py train "path/to/dataset" --model_type all --epochs 50 --batch_size 8 --num_frames 35 --num_classes 10 --output_dir "output"
# python script.py predict 'output/improved_model_best.pth' 'test_video.mp4'

import os
import sys

# Fix for MediaPipe threading issues
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Prevent threading conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

os.system("chcp 65001")  # Set console to UTF-8
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mediapipe as mp
import argparse
import pandas as pd
import warnings
import random
import time
from collections import Counter, defaultdict
from itertools import cycle
import json

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_holistic = mp.solutions.holistic

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ================== LABEL SMOOTHING FOR REGULARIZATION ==================
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing loss for better regularization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, dim=-1)
        loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1))
        loss = loss.squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        loss = (1 - self.smoothing) * loss + self.smoothing * smooth_loss
        return loss.mean()

# ================== ENHANCED KEYPOINT PROCESSORS ==================
class ImprovedKeypointProcessor:
    """Improved keypoint processing with better normalization"""
    
    @staticmethod
    def extract_pose_keypoints(landmarks):
        if landmarks:
            # Extended upper body indices for better representation
            upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 5, 7, 8]
            keypoints = []
            for i in upper_body_indices:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.visibility])
                else:
                    keypoints.extend([0, 0, 0])
            return np.array(keypoints)
        else:
            return np.zeros(14 * 3)
    
    @staticmethod
    def extract_hand_keypoints(landmarks):
        if landmarks:
            # Key hand points for better gesture recognition
            key_indices = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18]
            keypoints = []
            for i in key_indices:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0, 0, 0])
            return np.array(keypoints)
        else:
            return np.zeros(16 * 3)
    
    @staticmethod
    def normalize_keypoints(keypoints):
        """Improved normalization with robustness"""
        if len(keypoints) == 0:
            return keypoints
        
        normalized = keypoints.copy()
        
        # Per-frame normalization
        for frame_idx in range(normalized.shape[0]):
            frame = normalized[frame_idx]
            non_zero_mask = frame != 0
            
            if np.sum(non_zero_mask) > 10:
                non_zero_values = frame[non_zero_mask]
                mean_val = np.mean(non_zero_values)
                std_val = np.std(non_zero_values) + 1e-8
                normalized[frame_idx][non_zero_mask] = (frame[non_zero_mask] - mean_val) / std_val
        
        return normalized

class AdvancedKeypointProcessor:
    """Advanced processor with temporal and geometric features - FIXED"""
    
    @staticmethod
    def extract_pose_keypoints(landmarks):
        if landmarks:
            # Use same indices as improved for consistency
            key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 5, 7, 8]
            keypoints = []
            landmark_coords = []
            
            for i in key_indices:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.visibility])
                    landmark_coords.append([lm.x, lm.y])
                else:
                    keypoints.extend([0, 0, 0])
                    landmark_coords.append([0, 0])
            
            # Add geometric features (simplified)
            if len(landmark_coords) >= 2:
                # Shoulder width
                if landmark_coords[0] != [0, 0] and landmark_coords[1] != [0, 0]:
                    shoulder_width = np.linalg.norm(np.array(landmark_coords[0]) - np.array(landmark_coords[1]))
                    keypoints.append(shoulder_width)
                else:
                    keypoints.append(0)
            else:
                keypoints.append(0)
            
            return np.array(keypoints)
        else:
            return np.zeros(14 * 3 + 1)  # 43 features
    
    @staticmethod
    def extract_hand_keypoints(landmarks):
        if landmarks:
            # All hand keypoints
            keypoints = []
            for i in range(21):
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0, 0, 0])
            return np.array(keypoints)
        else:
            return np.zeros(21 * 3)  # 63 features
    
    @staticmethod
    def normalize_keypoints(keypoints):
        """Improved normalization with robustness"""
        if len(keypoints) == 0:
            return keypoints
        
        normalized = keypoints.copy()
        
        # Global normalization across all frames and features
        # This preserves temporal relationships
        non_zero_mask = normalized != 0
        if np.sum(non_zero_mask) > 10:
            non_zero_values = normalized[non_zero_mask]
            mean_val = np.mean(non_zero_values)
            std_val = np.std(non_zero_values) + 1e-8
            normalized[non_zero_mask] = (normalized[non_zero_mask] - mean_val) / std_val
        
        return normalized
    
    @staticmethod
    def compute_temporal_features(keypoints):
        """Compute temporal derivatives - FIXED to match expected size"""
        if len(keypoints) < 2:
            return keypoints
        
        # First derivative (velocity) with smaller weight
        velocity = np.diff(keypoints, axis=0)
        velocity = np.vstack([velocity[0:1], velocity])
        
        # Use only 50 additional features from velocity to get to 219
        # 169 base features + 50 velocity features = 219
        velocity_subset = velocity[:, :50]  # Take only first 50 features
        
        # Combine features
        enhanced_features = np.concatenate([keypoints, velocity_subset * 0.3], axis=1)
        
        return enhanced_features

# ================== ADAPTIVE DATASET CLASS ==================
class AdaptiveDataset(Dataset):
    """Dataset with adaptive handling for different numbers of classes"""
    
    def __init__(self, root_dir, num_classes=None, specific_classes=None, num_frames=30, 
                 split='train', test_size=0.2, processor_type='improved', 
                 min_samples_per_class=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.split = split
        self.augment = (split == 'train')
        self.processor_type = processor_type
        
        # Adaptive parameters based on number of classes
        if num_classes:
            self.num_classes = num_classes
            # Scale samples per class based on total classes
            if min_samples_per_class is None:
                self.min_samples_per_class = max(20, 100 // num_classes)
            else:
                self.min_samples_per_class = min_samples_per_class
                
            if max_samples_per_class is None:
                self.max_samples_per_class = max(60, 300 // num_classes)
            else:
                self.max_samples_per_class = max_samples_per_class
        else:
            self.num_classes = None
            self.min_samples_per_class = min_samples_per_class or 20
            self.max_samples_per_class = max_samples_per_class or 60
        
        # Set processor and feature size - FIXED
        # if processor_type == 'improved':
        #     self.processor = ImprovedKeypointProcessor()
        #     self.feature_size = 14*3 + 16*3 + 16*3  # 138
        # else:
        #     self.processor = AdvancedKeypointProcessor()
        #     self.feature_size = 43 + 63 + 63  # 169 (fixed calculation)
        #     self.temporal_feature_size = int(self.feature_size * 1.3)  # ~220
        if processor_type == 'improved':
            self.processor = ImprovedKeypointProcessor()
            self.feature_size = 14*3 + 16*3 + 16*3  # 138
        else:
            self.processor = AdvancedKeypointProcessor()
            self.feature_size = 43 + 63 + 63  # 169 (fixed calculation)
            # Fixed temporal feature size to match model expectation
            self.temporal_feature_size = 219  # 169 + 50 velocity features
        
        # Select classes
        all_classes = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
        
        if specific_classes:
            self.classes = [c.strip() for c in specific_classes if c.strip() in all_classes]
        elif num_classes:
            self.classes = self._select_best_classes(all_classes, num_classes)
        else:
            self.classes = self._select_best_classes(all_classes, 8)
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Load samples with adaptive strategy
        self.samples = self._load_adaptive_samples()
        self._apply_balanced_split(test_size)
        
        print(f"\nDataset ({processor_type}) - {split}:")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Min samples per class: {self.min_samples_per_class}")
        print(f"  Max samples per class: {self.max_samples_per_class}")
        
        # Print class distribution
        class_dist = Counter([s[1] for s in self.samples])
        print(f"  Class distribution:")
        for class_idx in sorted(class_dist.keys()):
            class_name = self.idx_to_class.get(class_idx, f"Class_{class_idx}")
            count = class_dist[class_idx]
            print(f"    {class_name}: {count} samples")
    
    def _select_best_classes(self, all_classes, num_classes):
        """Select best quality classes with sufficient samples"""
        class_quality = []
        
        print(f"Evaluating classes for selection (target: {num_classes} classes)...")
        
        for cls in tqdm(all_classes, desc="Scanning classes"):
            cls_dir = os.path.join(self.root_dir, cls)
            if os.path.exists(cls_dir):
                video_files = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')]
                
                # Quick quality check on subset
                quality_count = 0
                check_count = min(50, len(video_files))
                
                for video_file in random.sample(video_files, min(check_count, len(video_files))):
                    video_path = os.path.join(cls_dir, video_file)
                    if self._is_quality_video(video_path):
                        quality_count += 1
                
                # Estimate total quality videos
                estimated_quality = (quality_count / check_count) * len(video_files) if check_count > 0 else 0
                
                if estimated_quality >= self.min_samples_per_class:
                    class_quality.append((cls, estimated_quality, len(video_files)))
        
        # Sort by quality count
        class_quality.sort(key=lambda x: x[1], reverse=True)
        
        # Select top classes
        selected_classes = [cls for cls, _, _ in class_quality[:num_classes]]
        
        print(f"Selected {len(selected_classes)} classes:")
        for cls, quality, total in class_quality[:num_classes]:
            print(f"  {cls}: ~{int(quality)} quality videos out of {total} total")
        
        return selected_classes
    
    def _is_quality_video(self, video_path):
        """Quick quality check for videos"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adaptive quality thresholds based on number of classes
            min_frames = 10 if self.num_classes and self.num_classes > 8 else 15
            
            if (frame_count < min_frames or fps < 10 or width < 160 or height < 160):
                cap.release()
                return False
            
            # Quick frame check
            valid_frames = 0
            check_points = [0, frame_count//2, frame_count-1]
            
            for i in check_points:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, frame_count-1))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 5000:
                    valid_frames += 1
            
            cap.release()
            return valid_frames >= 2
        except:
            return False
    
    def _load_adaptive_samples(self):
        """Load samples with adaptive strategy based on number of classes"""
        samples = []
        
        print(f"Loading samples with adaptive strategy...")
        
        for cls in tqdm(self.classes, desc="Loading classes"):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            
            video_files = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')]
            class_samples = []
            
            # Check more videos for larger class counts
            check_limit = self.max_samples_per_class * 2
            
            for video_file in video_files[:check_limit]:
                video_path = os.path.join(cls_dir, video_file)
                if self._is_quality_video(video_path):
                    class_samples.append((video_path, self.class_to_idx[cls]))
                
                if len(class_samples) >= self.max_samples_per_class:
                    break
            
            # Apply sampling strategy
            if len(class_samples) > self.max_samples_per_class:
                class_samples = random.sample(class_samples, self.max_samples_per_class)
            elif len(class_samples) < self.min_samples_per_class and len(class_samples) > 0:
                # Duplicate samples if too few (with warning)
                print(f"  Warning: {cls} has only {len(class_samples)} samples, duplicating to reach {self.min_samples_per_class}")
                while len(class_samples) < self.min_samples_per_class:
                    class_samples.extend(class_samples[:min(len(class_samples), 
                                                            self.min_samples_per_class - len(class_samples))])
            
            samples.extend(class_samples)
        
        return samples
    
    def _apply_balanced_split(self, test_size):
        """Apply train/val split with minimum samples per class"""
        if len(self.samples) == 0:
            raise ValueError("No valid videos found")
        
        # Group samples by class
        class_samples = defaultdict(list)
        for sample in self.samples:
            class_samples[sample[1]].append(sample)
        
        train_samples = []
        val_samples = []
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Adaptive validation size based on number of classes
        min_val_per_class = 2 if len(self.classes) <= 6 else 3
        
        for class_idx, samples in class_samples.items():
            random.shuffle(samples)
            n_samples = len(samples)
            
            if n_samples < min_val_per_class * 2:
                # Too few samples - use all for training and duplicate for validation
                if self.split == 'train':
                    train_samples.extend(samples)
                else:
                    # Ensure minimum validation samples
                    val_samples.extend(samples[:min_val_per_class])
            else:
                # Calculate validation size
                n_val = max(min_val_per_class, int(n_samples * test_size))
                n_val = min(n_val, n_samples // 3)  # Don't take more than 1/3 for validation
                
                if self.split == 'train':
                    train_samples.extend(samples[n_val:])
                else:
                    val_samples.extend(samples[:n_val])
        
        if self.split == 'train':
            self.samples = train_samples
            random.shuffle(self.samples)
        else:
            self.samples = val_samples
    
    def get_class_weights(self):
        """Get class weights for loss balancing"""
        class_counts = Counter([s[1] for s in self.samples])
        total_samples = len(self.samples)
        weights = {cls: total_samples / (len(self.classes) * count) 
                  for cls, count in class_counts.items()}
        return torch.tensor([weights.get(i, 1.0) for i in range(len(self.classes))], dtype=torch.float)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        keypoints = self._extract_keypoints(video_path)
        
        # Adaptive augmentation based on number of classes
        augment_prob = 0.3 + (0.05 * max(0, len(self.classes) - 5))  # Increase for more classes
        if self.augment and random.random() < augment_prob:
            keypoints = self._adaptive_augment(keypoints)
        
        return torch.tensor(keypoints, dtype=torch.float32), label
    
    def _extract_keypoints(self, video_path):
        """Extract keypoints with error handling - FIXED"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints = []
        
        # Use consistent model complexity
        model_complexity = 1  # Always use 1 for consistency
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
            model_complexity=model_complexity
        ) as holistic:
            
            if total_frames <= self.num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    keypoints.append(np.zeros(self.feature_size))
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                pose = self.processor.extract_pose_keypoints(results.pose_landmarks)
                lh = self.processor.extract_hand_keypoints(results.left_hand_landmarks)
                rh = self.processor.extract_hand_keypoints(results.right_hand_landmarks)
                
                frame_features = np.concatenate([pose, lh, rh])
                
                # Ensure consistent feature size
                if len(frame_features) != self.feature_size:
                    if len(frame_features) < self.feature_size:
                        padding = np.zeros(self.feature_size - len(frame_features))
                        frame_features = np.concatenate([frame_features, padding])
                    else:
                        frame_features = frame_features[:self.feature_size]
                
                keypoints.append(frame_features)
        
        cap.release()
        
        # Ensure correct number of frames
        while len(keypoints) < self.num_frames:
            if keypoints:
                keypoints.append(keypoints[-1])
            else:
                keypoints.append(np.zeros(self.feature_size))
        
        keypoints = np.array(keypoints[:self.num_frames])
        
        # FIXED ORDER: Compute temporal features FIRST, then normalize
        if self.processor_type == 'advanced':
            keypoints = self.processor.compute_temporal_features(keypoints)
            # Update temporal feature size after computation
            self.temporal_feature_size = keypoints.shape[1]
        
        # Normalize AFTER temporal features
        keypoints = self.processor.normalize_keypoints(keypoints)
        
        return keypoints
    
    def _adaptive_augment(self, keypoints):
        """Adaptive augmentation based on class count"""
        augmented = keypoints.copy()
        
        # More aggressive augmentation for more classes
        noise_scale = 0.005 * (1 + 0.1 * len(self.classes))
        
        if random.random() < 0.5:
            noise = np.random.normal(0, noise_scale, augmented.shape)
            augmented = augmented + noise
        
        if random.random() < 0.3 and len(keypoints) > 8:
            shift = random.randint(-2, 2)
            if shift > 0:
                augmented = np.vstack([keypoints[shift:], keypoints[-shift:]])
            elif shift < 0:
                augmented = np.vstack([keypoints[:shift], keypoints[:-shift]])
        
        # Random scaling for more classes
        if len(self.classes) > 6 and random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            augmented = augmented * scale
        
        return augmented

# ================== ADAPTIVE MODEL ARCHITECTURES ==================
class AdaptiveImprovedCNNLSTM(nn.Module):
    """Improved model with adaptive architecture based on number of classes"""
    
    def __init__(self, num_classes, input_size=138, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Scale architecture based on number of classes
        scaling_factor = min(2.0, 1.0 + (num_classes - 5) * 0.15)
        
        hidden_1 = int(128 * scaling_factor)
        hidden_2 = int(96 * scaling_factor)
        hidden_3 = int(64 * scaling_factor)
        lstm_hidden = int(160 * scaling_factor)
        
        # Adjust dropout based on class count
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
            nn.Dropout(dropout_rate * 0.6)
        )
        
        # More LSTM layers for more classes
        num_lstm_layers = 2 if num_classes > 8 else 1
        self.lstm = nn.LSTM(hidden_3, lstm_hidden, batch_first=True, 
                           dropout=dropout_rate if num_lstm_layers > 1 else 0, 
                           num_layers=num_lstm_layers)
        
        # Adaptive classifier
        classifier_hidden = int(96 * scaling_factor)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(classifier_hidden, int(classifier_hidden * 0.5)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(int(classifier_hidden * 0.5), num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Better initialization for more classes
                if self.num_classes > 8:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        x_reshaped = x.view(batch_size * seq_len, features)
        features_out = self.feature_layers(x_reshaped)
        features_out = features_out.view(batch_size, seq_len, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(features_out)
        
        # Use last hidden state
        if self.lstm.num_layers > 1:
            final_features = h_n[-1]
        else:
            final_features = h_n[0]
        
        return self.classifier(final_features)

# class AdaptiveAdvancedCNNLSTM(nn.Module):
#     """Advanced model with attention and adaptive architecture - FIXED"""
    
#     def __init__(self, num_classes, input_size=None, dropout_rate=0.35):
#         super().__init__()
#         self.num_classes = num_classes
        
#         # Auto-detect input size if not provided
#         if input_size is None:
#             # Base: 43 (pose) + 63 (left hand) + 63 (right hand) = 169
#             # With temporal: 169 * 1.3 = ~220
#             input_size = 220
        
#         self.input_size = input_size
        
#         # Reduce model complexity - less aggressive scaling
#         scaling_factor = min(1.5, 1.0 + (num_classes - 5) * 0.1)
        
#         hidden_1 = int(128 * scaling_factor)
#         hidden_2 = int(96 * scaling_factor)
#         hidden_3 = int(64 * scaling_factor)
#         lstm_hidden = int(128 * scaling_factor)
        
#         # Reduce dropout for advanced model
#         dropout_rate = dropout_rate * 0.8
        
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_size, hidden_1),
#             nn.BatchNorm1d(hidden_1),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate * 0.5),
            
#             nn.Linear(hidden_1, hidden_2),
#             nn.BatchNorm1d(hidden_2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate * 0.6),
            
#             nn.Linear(hidden_2, hidden_3),
#             nn.BatchNorm1d(hidden_3),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate * 0.7)
#         )
        
#         # Reduce LSTM layers - 2 max
#         num_lstm_layers = 2 if num_classes > 8 else 1
#         self.lstm = nn.LSTM(
#             input_size=hidden_3,
#             hidden_size=lstm_hidden,
#             num_layers=num_lstm_layers,
#             batch_first=True,
#             dropout=dropout_rate * 0.5 if num_lstm_layers > 1 else 0,
#             bidirectional=True
#         )
        
#         # Simpler attention - reduce heads
#         num_heads = 4 if num_classes <= 8 else 8
#         self.attention = nn.MultiheadAttention(
#             embed_dim=lstm_hidden * 2,
#             num_heads=num_heads,
#             dropout=dropout_rate * 0.3,
#             batch_first=True
#         )
        
#         # Simpler classifier
#         classifier_hidden = int(128 * scaling_factor)
#         self.classifier = nn.Sequential(
#             nn.Linear(lstm_hidden * 2, classifier_hidden),
#             nn.BatchNorm1d(classifier_hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate * 0.8),
            
#             nn.Linear(classifier_hidden, num_classes)
#         )
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LSTM):
#                 for name, param in m.named_parameters():
#                     if 'weight_ih' in name:
#                         nn.init.xavier_uniform_(param.data)
#                     elif 'weight_hh' in name:
#                         nn.init.orthogonal_(param.data)
#                     elif 'bias' in name:
#                         nn.init.constant_(param.data, 0)
    
#     def forward(self, x):
#         batch_size, seq_len, features = x.shape
        
#         x_reshaped = x.view(batch_size * seq_len, features)
#         features_out = self.feature_extractor(x_reshaped)
#         features_out = features_out.view(batch_size, seq_len, -1)
        
#         lstm_out, (h_n, c_n) = self.lstm(features_out)
        
#         # Self-attention
#         attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
#         # Global average pooling
#         attended_features = torch.mean(attn_out, dim=1)
        
#         return self.classifier(attended_features)
class AdaptiveAdvancedCNNLSTM(nn.Module):
    """Advanced model with attention and adaptive architecture - FIXED"""
    
    def __init__(self, num_classes, input_size=None, dropout_rate=0.35):
        super().__init__()
        self.num_classes = num_classes
        
        # Fixed input size for consistency
        if input_size is None:
            input_size = 219  # Fixed: 169 base + 50 temporal
        
        self.input_size = input_size
        
        # Reduce model complexity - less aggressive scaling
        scaling_factor = min(1.5, 1.0 + (num_classes - 5) * 0.1)
        
        hidden_1 = int(128 * scaling_factor)
        hidden_2 = int(96 * scaling_factor)
        hidden_3 = int(64 * scaling_factor)
        lstm_hidden = int(128 * scaling_factor)
        
        # Reduce dropout for advanced model
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
            nn.Dropout(dropout_rate * 0.7)
        )
        
        # Reduce LSTM layers - 2 max
        num_lstm_layers = 2 if num_classes > 8 else 1
        self.lstm = nn.LSTM(
            input_size=hidden_3,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate * 0.5 if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Simpler attention - reduce heads
        num_heads = 4 if num_classes <= 8 else 8
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout_rate * 0.3,
            batch_first=True
        )
        
        # Simpler classifier
        classifier_hidden = int(128 * scaling_factor)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(classifier_hidden, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        x_reshaped = x.view(batch_size * seq_len, features)
        features_out = self.feature_extractor(x_reshaped)
        features_out = features_out.view(batch_size, seq_len, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(features_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        attended_features = torch.mean(attn_out, dim=1)
        
        return self.classifier(attended_features)

class AdaptiveEnsembleModel(nn.Module):
    """Ensemble model with adaptive weighting"""
    
    def __init__(self, num_classes, input_size_1=138, input_size_2=220):
        super().__init__()
        
        self.model1 = AdaptiveImprovedCNNLSTM(num_classes, input_size_1, dropout_rate=0.3)
        self.model2 = AdaptiveAdvancedCNNLSTM(num_classes, input_size_2, dropout_rate=0.35)
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Optional: meta-learner for dynamic weighting
        self.use_meta_learner = num_classes > 8
        if self.use_meta_learner:
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x1, x2):
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        
        if self.use_meta_learner:
            # Dynamic weighting based on predictions
            combined = torch.cat([out1, out2], dim=1)
            weights = self.meta_learner(combined)
            ensemble_out = weights[:, 0:1] * out1 + weights[:, 1:2] * out2
        else:
            # Static learnable weights
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_out = weights[0] * out1 + weights[1] * out2
        
        return ensemble_out, out1, out2

# ================== TRAINING FUNCTIONS WITH ADAPTIVE STRATEGIES ==================
def get_adaptive_training_params(num_classes, base_lr=0.001, base_epochs=50, base_batch_size=8, model_type='improved'):
    """Get adaptive training parameters based on number of classes"""
    
    # Adjust learning rate
    if num_classes <= 5:
        lr = base_lr
    elif num_classes <= 8:
        lr = base_lr * 0.8
    elif num_classes <= 10:
        lr = base_lr * 0.6
    else:
        lr = base_lr * 0.5
    
    # Special adjustment for advanced model
    if model_type == 'advanced':
        lr = lr * 0.7  # Lower learning rate for advanced model
    
    # Adjust epochs
    if num_classes <= 5:
        epochs = base_epochs
    elif num_classes <= 8:
        epochs = int(base_epochs * 1.2)
    elif num_classes <= 10:
        epochs = int(base_epochs * 1.5)
    else:
        epochs = int(base_epochs * 2)
    
    # Adjust batch size
    if num_classes <= 5:
        batch_size = base_batch_size
    elif num_classes <= 8:
        batch_size = max(4, base_batch_size - 2)
    else:
        batch_size = max(4, base_batch_size // 2)
    
    # Patience for early stopping
    patience = 15 + num_classes
    
    return {
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience
    }

def train_single_model(args, model_type='improved'):
    """Train a single model with adaptive strategies"""
    # GPU/CPU detection and setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Print device information
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*60}")
    if device.type == 'cuda':
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print(f" Using CPU (GPU not available or disabled)")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Determine number of classes
    if args.classes:
        num_classes = len(args.classes.split(','))
    else:
        num_classes = args.num_classes
    
    # Get adaptive parameters
    adaptive_params = get_adaptive_training_params(
        num_classes, 
        base_lr=args.lr,
        base_epochs=args.epochs,
        base_batch_size=args.batch_size,
        model_type=model_type
    )
    
    print(f"Adaptive parameters for {num_classes} classes:")
    print(f"  Learning rate: {adaptive_params['lr']:.6f}")
    print(f"  Epochs: {adaptive_params['epochs']}")
    print(f"  Batch size: {adaptive_params['batch_size']}")
    print(f"  Patience: {adaptive_params['patience']}")
    
    # Create datasets
    train_dataset = AdaptiveDataset(
        args.data_dir,
        num_classes=num_classes,
        specific_classes=args.classes.split(',') if args.classes else None,
        num_frames=args.num_frames,
        split='train',
        test_size=args.test_size,
        processor_type=model_type
    )
    
    val_dataset = AdaptiveDataset(
        args.data_dir,
        specific_classes=train_dataset.classes,
        num_frames=args.num_frames,
        split='val',
        test_size=args.test_size,
        processor_type=model_type
    )
    
    # Get class weights for balanced training
    class_weights = train_dataset.get_class_weights().to(device)
    
    # Create data loaders with safer settings for MediaPipe
    num_workers = 0  # Changed to 0 to avoid threading issues
    pin_memory = True if device.type == 'cuda' else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=adaptive_params['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=adaptive_params['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    # Create model with adaptive architecture - FIXED
    if model_type == 'improved':
        model = AdaptiveImprovedCNNLSTM(
            len(train_dataset.classes), 
            input_size=train_dataset.feature_size, 
            dropout_rate=0.3
        )
    else:  # advanced
        # FIXED: Calculate actual feature size
        if hasattr(train_dataset, 'temporal_feature_size'):
            actual_input_size = train_dataset.temporal_feature_size
        else:
            # Get actual temporal feature size from a sample
            sample_input, _ = train_dataset[0]
            #actual_input_size = sample_input.shape[-1]  # Get the last dimension
            actual_input_size = 219  # Fixed size: 169 base + 50 temporal

        
        print(f"Detected input size for advanced model: {actual_input_size}")
        
        model = AdaptiveAdvancedCNNLSTM(
            len(train_dataset.classes), 
            input_size=actual_input_size,
            dropout_rate=0.35  # Reduced from 0.4
        )
    
    model = model.to(device)
    
    # Print model summary and save architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Visualize model architecture
    if model_type == 'advanced':
        sample_input_shape = (adaptive_params['batch_size'], args.num_frames, actual_input_size)
    else:
        sample_input_shape = (adaptive_params['batch_size'], args.num_frames, train_dataset.feature_size)
    visualize_model_architecture(model, f"{model_type.capitalize()}_CNNLSTM", args.output_dir, sample_input_shape)
    
    # Training setup with class weights
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=adaptive_params['lr'], 
        weight_decay=5e-4
    )
    
    # Use label smoothing for advanced model
    if model_type == 'advanced':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Adaptive scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=8, 
        factor=0.7, 
        min_lr=1e-6
    )
    
    # Training variables - FIXED LOGIC FOR BEST MODEL
    best_train_acc = 0.0  # Track highest training accuracy
    best_val_acc = 0.0
    best_f1 = 0.0
    best_gap = float('inf')  # Track the gap for the best model
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # Track best epoch predictions for final metrics
    best_y_true = []
    best_y_pred = []
    
    start_time = time.time()
    
    for epoch in range(adaptive_params['epochs']):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{adaptive_params['epochs']} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L2 regularization for many classes
            if num_classes > 8:
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # More aggressive gradient clipping for advanced model
            if model_type == 'advanced':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.2%}'
            })
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred, y_scores = [], [], []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{adaptive_params['epochs']} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Use same criterion for validation
                if model_type == 'advanced':
                    loss = criterion(outputs, labels)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, labels)  # No class weights for validation
                
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.argmax(1).cpu().numpy())
                y_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_correct/val_total:.2%}'
                })
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\nEpoch {epoch+1}/{adaptive_params['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.3f}")
        print(f"Accuracy Gap: {abs(train_acc - val_acc):.2%}")
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Calculate accuracy gap
        accuracy_gap = abs(train_acc - val_acc)
        
        # NEW LOGIC: Save model with highest training accuracy where gap <= 10%
        if accuracy_gap <= 0.10:  # Gap is within 10%
            should_save = False
            
            if train_acc > best_train_acc:
                # Higher training accuracy with acceptable gap
                should_save = True
                save_reason = "Higher training accuracy"
            elif train_acc == best_train_acc and accuracy_gap < best_gap:
                # Same training accuracy but smaller gap
                should_save = True
                save_reason = "Same training accuracy, smaller gap"
            
            if should_save:
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_f1 = val_f1
                best_gap = accuracy_gap
                patience_counter = 0
                
                # Store best predictions for final metrics
                best_y_true = y_true.copy()
                best_y_pred = y_pred.copy()
                
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Save the model
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'classes': train_dataset.classes,
                    'class_to_idx': train_dataset.class_to_idx,
                    'num_frames': args.num_frames,
                    'model_type': model_type,
                    'num_classes': len(train_dataset.classes),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'accuracy_gap': accuracy_gap,
                    'val_f1': val_f1,
                    'precision': precision,
                    'recall': recall,
                    'epoch': epoch,
                    'history': history,
                    'input_size': model.input_size if hasattr(model, 'input_size') else train_dataset.feature_size
                }, f"{args.output_dir}/{model_type}_model_best.pth")
                
                print(f"✓ New best model saved! ({save_reason})")
                print(f"  Train: {train_acc:.2%} | Val: {val_acc:.2%} | Gap: {accuracy_gap:.2%}")
                
                # Save visualizations for best model
                plot_comprehensive_results(
                    history, y_true, y_pred, np.array(y_scores), 
                    train_dataset.classes, args.output_dir, f"{model_type.capitalize()}_CNNLSTM"
                )
                
                plot_confusion_matrix(
                    y_true, y_pred, train_dataset.classes,
                    f"{args.output_dir}/{model_type}_confusion_matrix.png",
                    f"{model_type.capitalize()}_CNNLSTM"
                )
        else:
            patience_counter += 1
            print(f"⚠ Accuracy gap too large: {accuracy_gap:.2%} > 10% - Model not saved")
        
        # Early stopping
        if patience_counter >= adaptive_params['patience']:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Clear GPU cache periodically to prevent memory issues
        if device.type == 'cuda' and epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        print("-" * 50)
    
    training_time = (time.time() - start_time) / 60
    
    print(f"\n{'='*50}")
    print(f"{model_type.upper()} MODEL TRAINING COMPLETED!")
    print(f"{'='*50}")
    print(f"Classes: {len(train_dataset.classes)}")
    print(f"Best Model - Train Acc: {best_train_acc:.2%} | Val Acc: {best_val_acc:.2%} | Gap: {best_gap:.2%} | F1: {best_f1:.3f}")
    print(f"Training Time: {training_time:.1f} minutes")
    
    # Return metrics for the best model
    if len(best_y_pred) > 0:
        final_precision = precision_score(best_y_true, best_y_pred, average='weighted', zero_division=0)
        final_recall = recall_score(best_y_true, best_y_pred, average='weighted', zero_division=0)
    else:
        final_precision = 0.0
        final_recall = 0.0
    
    final_metrics = {
        'accuracy': best_val_acc,  # Return validation accuracy
        'train_accuracy': best_train_acc,  # Also return training accuracy
        'f1_score': best_f1,
        'precision': final_precision,
        'recall': final_recall,
        'training_time': training_time,
        'accuracy_gap': best_gap,
        'num_classes': len(train_dataset.classes)
    }
    
    return final_metrics, train_dataset.classes

def train_ensemble_model(args):
    """Train ensemble model with adaptive strategies"""
    # GPU/CPU detection and setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Print device information
    print(f"\n{'='*60}")
    print(f"Training ENSEMBLE model")
    print(f"{'='*60}")
    if device.type == 'cuda':
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Available Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
    else:
        print(f" Using CPU (GPU not available or disabled)")
    print(f"{'='*60}")
    
    # Determine number of classes
    if args.classes:
        num_classes = len(args.classes.split(','))
    else:
        num_classes = args.num_classes
    
    # Get adaptive parameters
    adaptive_params = get_adaptive_training_params(
        num_classes, 
        base_lr=args.lr * 0.8,  # Slightly lower for ensemble
        base_epochs=args.epochs,
        base_batch_size=max(4, args.batch_size - 2)  # Smaller batch for memory
    )
    
    # Create datasets for both processors
    train_dataset_1 = AdaptiveDataset(
        args.data_dir,
        num_classes=num_classes,
        specific_classes=args.classes.split(',') if args.classes else None,
        num_frames=args.num_frames,
        split='train',
        test_size=args.test_size,
        processor_type='improved'
    )
    
    train_dataset_2 = AdaptiveDataset(
        args.data_dir,
        specific_classes=train_dataset_1.classes,
        num_frames=args.num_frames,
        split='train',
        test_size=args.test_size,
        processor_type='advanced'
    )
    
    val_dataset_1 = AdaptiveDataset(
        args.data_dir,
        specific_classes=train_dataset_1.classes,
        num_frames=args.num_frames,
        split='val',
        test_size=args.test_size,
        processor_type='improved'
    )
    
    val_dataset_2 = AdaptiveDataset(
        args.data_dir,
        specific_classes=train_dataset_1.classes,
        num_frames=args.num_frames,
        split='val',
        test_size=args.test_size,
        processor_type='advanced'
    )
    
    # Create ensemble model - FIXED
    # Get actual input sizes from samples
    sample1, _ = train_dataset_1[0]
    sample2, _ = train_dataset_2[0]
    input_size_1 = sample1.shape[-1]
    input_size_2 = sample2.shape[-1]
    
    print(f"\nEnsemble model input sizes - Model 1: {input_size_1}, Model 2: {input_size_2}")
    
    model = AdaptiveEnsembleModel(
        len(train_dataset_1.classes),
        input_size_1=input_size_1,
        input_size_2=input_size_2
    ).to(device)
    
    # Save model architecture
    sample_input_shape = (adaptive_params['batch_size'], args.num_frames, input_size_1)
    visualize_model_architecture(model, "Ensemble_Model", args.output_dir, sample_input_shape)
    
    # Get class weights
    class_weights = train_dataset_1.get_class_weights().to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=adaptive_params['lr'], weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.7, min_lr=1e-6
    )
    
    # Training variables - FIXED LOGIC
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_f1 = 0.0
    best_gap = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # Track best epoch predictions for final metrics
    best_y_true = []
    best_y_pred = []
    
    start_time = time.time()
    
    for epoch in range(adaptive_params['epochs']):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Combine datasets
        combined_data = list(zip(train_dataset_1, train_dataset_2))
        
        # Safer settings for data loading
        num_workers = 0  # No multiprocessing to avoid MediaPipe issues
        pin_memory = True if device.type == 'cuda' else False
        
        train_loader = DataLoader(
            combined_data, 
            batch_size=adaptive_params['batch_size'], 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{adaptive_params['epochs']} [Train]")
        for (inputs1, labels1), (inputs2, labels2) in train_bar:
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            labels = labels1.to(device)
            
            optimizer.zero_grad()
            ensemble_out, _, _ = model(inputs1, inputs2)
            loss = criterion(ensemble_out, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (ensemble_out.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.2%}'
            })
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred, y_scores = [], [], []
        
        combined_val_data = list(zip(val_dataset_1, val_dataset_2))
        
        val_loader = DataLoader(
            combined_val_data, 
            batch_size=adaptive_params['batch_size'], 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{adaptive_params['epochs']} [Val]")
        with torch.no_grad():
            for (inputs1, labels1), (inputs2, labels2) in val_bar:
                inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
                labels = labels1.to(device)
                
                ensemble_out, _, _ = model(inputs1, inputs2)
                loss = criterion(ensemble_out, labels)
                
                val_loss += loss.item()
                val_correct += (ensemble_out.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(ensemble_out.argmax(1).cpu().numpy())
                y_scores.extend(F.softmax(ensemble_out, dim=1).cpu().numpy())
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_correct/val_total:.2%}'
                })
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\nEpoch {epoch+1}/{adaptive_params['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.3f}")
        
        scheduler.step(val_loss)

        # Calculate accuracy gap
        accuracy_gap = abs(train_acc - val_acc)
        print(f"Accuracy Gap: {accuracy_gap:.2%}")
        
        # NEW LOGIC: Save model with highest training accuracy where gap <= 10%
        if accuracy_gap <= 0.10:  # Gap is within 10%
            should_save = False
            
            if train_acc > best_train_acc:
                # Higher training accuracy with acceptable gap
                should_save = True
                save_reason = "Higher training accuracy"
            elif train_acc == best_train_acc and accuracy_gap < best_gap:
                # Same training accuracy but smaller gap
                should_save = True
                save_reason = "Same training accuracy, smaller gap"
            
            if should_save:
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_f1 = val_f1
                best_gap = accuracy_gap
                patience_counter = 0
                
                # Save best predictions for final metrics
                best_y_true = y_true.copy()
                best_y_pred = y_pred.copy()
                
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                
                torch.save({
                    'model_state': model.state_dict(),
                    'classes': train_dataset_1.classes,
                    'class_to_idx': train_dataset_1.class_to_idx,
                    'num_frames': args.num_frames,
                    'model_type': 'ensemble',
                    'num_classes': len(train_dataset_1.classes),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'accuracy_gap': accuracy_gap,
                    'val_f1': val_f1,
                    'precision': precision,
                    'recall': recall,
                    'epoch': epoch,
                    'history': history,
                    'input_size_1': input_size_1,
                    'input_size_2': input_size_2
                }, f"{args.output_dir}/ensemble_model_best.pth")
                
                print(f"✓ New best ensemble model! ({save_reason})")
                print(f"  Train: {train_acc:.2%} | Val: {val_acc:.2%} | Gap: {accuracy_gap:.2%}")
                
                # Save visualizations
                plot_comprehensive_results(
                    history, y_true, y_pred, np.array(y_scores), 
                    train_dataset_1.classes, args.output_dir, "Ensemble"
                )
                
                plot_confusion_matrix(
                    y_true, y_pred, train_dataset_1.classes,
                    f"{args.output_dir}/ensemble_confusion_matrix.png",
                    "Ensemble"
                )
        else:
            patience_counter += 1
            print(f"⚠ Accuracy gap too large: {accuracy_gap:.2%} > 10% - Model not saved")
        
        # Early stopping
        if patience_counter >= adaptive_params['patience']:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Clear GPU cache periodically to prevent memory issues
        if device.type == 'cuda' and epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        print("-" * 50)
    
    training_time = (time.time() - start_time) / 60
    
    print(f"\n{'='*50}")
    print(f"ENSEMBLE MODEL TRAINING COMPLETED!")
    print(f"Best Model - Train Acc: {best_train_acc:.2%} | Val Acc: {best_val_acc:.2%} | Gap: {best_gap:.2%} | F1: {best_f1:.3f}")
    print(f"Training Time: {training_time:.1f} minutes")
    
    # Return metrics with all required fields
    if len(best_y_pred) > 0:
        final_precision = precision_score(best_y_true, best_y_pred, average='weighted', zero_division=0)
        final_recall = recall_score(best_y_true, best_y_pred, average='weighted', zero_division=0)
    else:
        final_precision = 0.0
        final_recall = 0.0
    
    final_metrics = {
        'accuracy': best_val_acc,
        'train_accuracy': best_train_acc,
        'f1_score': best_f1,
        'precision': final_precision,
        'recall': final_recall,
        'training_time': training_time,
        'accuracy_gap': best_gap,
        'num_classes': len(train_dataset_1.classes)
    }
    
    return final_metrics, train_dataset_1.classes

def train_all_models(args):
    """Train all models and compare"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    print("="*60)
    print("TRAINING ALL MODELS WITH ADAPTIVE STRATEGIES")
    print(f"Target classes: {args.num_classes if not args.classes else len(args.classes.split(','))}")
    print("="*60)
    
    # Train models
    print("\n1. Training Improved CNN-LSTM Model...")
    improved_metrics, classes = train_single_model(args, 'improved')
    results['Improved'] = improved_metrics
    
    print("\n2. Training Advanced CNN-LSTM Model...")
    advanced_metrics, _ = train_single_model(args, 'advanced')
    results['Advanced'] = advanced_metrics
    
    print("\n3. Training Ensemble Model...")
    ensemble_metrics, _ = train_ensemble_model(args)
    results['Ensemble'] = ensemble_metrics
    
    # Save comparison
    results_df = pd.DataFrame(results).transpose()
    results_df.to_csv(f'{args.output_dir}/model_comparison.csv')
    
    # Create comparison plot
    compare_models(results, args.output_dir)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Train Acc':<12} {'Val Acc':<10} {'Gap':<8} {'F1-Score':<10} {'Time (min)':<12}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['train_accuracy']:<12.3f} {metrics['accuracy']:<10.3f} "
              f"{metrics['accuracy_gap']:<8.3f} {metrics['f1_score']:<10.3f} "
              f"{metrics['training_time']:<12.1f}")
    
    print(f"\n✓ Model comparison plot saved to {args.output_dir}/model_comparison.png")
    
    return results

# ================== VISUALIZATION FUNCTIONS ==================
def plot_comprehensive_results(history, y_true, y_pred, y_scores, classes, output_dir, model_name):
    """Create comprehensive result visualizations"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training History (Loss and Accuracy)
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(history['train_loss'], 'o-', label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], 'o-', label='Val Loss', linewidth=2)
    plt.title(f'{model_name} - Training Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(history['train_acc'], 'o-', label='Train Acc', linewidth=2)
    plt.plot(history['val_acc'], 'o-', label='Val Acc', linewidth=2)
    plt.title(f'{model_name} - Training Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. F1 Score
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(history['val_f1'], 'o-', label='Val F1', color='green', linewidth=2)
    plt.title(f'{model_name} - F1 Score', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax4 = plt.subplot(3, 3, 4)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=ax4)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 4. Per-Class F1 Scores
    ax5 = plt.subplot(3, 3, 5)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    bars = plt.bar(range(len(classes)), class_f1, color='skyblue', alpha=0.8)
    plt.title(f'{model_name} - Per-Class F1 Scores', fontsize=12, fontweight='bold')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(classes)), classes, rotation=45)
    
    # Add value labels on bars
    for bar, f1_val in zip(bars, class_f1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Per-Class Precision and Recall
    ax6 = plt.subplot(3, 3, 6)
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    x_pos = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x_pos - width/2, class_precision, width, label='Precision', alpha=0.8)
    plt.bar(x_pos + width/2, class_recall, width, label='Recall', alpha=0.8)
    
    plt.title(f'{model_name} - Precision & Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.xticks(x_pos, classes, rotation=45)
    plt.legend()
    
    # 6. ROC Curves (Multi-class)
    if y_scores is not None and len(classes) > 2:
        ax7 = plt.subplot(3, 3, 7)
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(len(classes)))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(classes)):
            if i < y_true_bin.shape[1] and i < len(y_scores[0]):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
        for i, color in zip(range(len(classes)), colors):
            if i in fpr:
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves', fontsize=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8)
    
    # 7. Classification Report Heatmap
    ax8 = plt.subplot(3, 3, 8)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Extract metrics for heatmap
    metrics_data = []
    for class_name in classes:
        if class_name in report:
            metrics_data.append([
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ])
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data, 
                                 columns=['Precision', 'Recall', 'F1-Score'],
                                 index=classes)
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax8)
        plt.title(f'{model_name} - Classification Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Classes')
    
    # 8. Model Comparison Summary
    ax9 = plt.subplot(3, 3, 9)
    overall_metrics = {
        'Accuracy': np.mean(np.array(y_true) == np.array(y_pred)),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Macro Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())
    
    bars = plt.bar(metric_names, metric_values, color='lightcoral', alpha=0.8)
    plt.title(f'{model_name} - Overall Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name.lower()}_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed metrics to CSV
    detailed_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(detailed_report).transpose()
    report_df.to_csv(f'{output_dir}/{model_name.lower()}_detailed_metrics.csv')
    
    return overall_metrics

def compare_models(results_dict, output_dir):
    """Compare multiple models side by side with enhanced metrics"""
    
    # Check if we have models to compare
    if not results_dict:
        print("No models to compare!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(results_dict.keys())
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    # 1. Training vs Validation Accuracy Comparison
    train_accs = [results_dict[model].get('train_accuracy', 0) for model in model_names]
    val_accs = [results_dict[model].get('accuracy', 0) for model in model_names]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, train_accs, width, label='Train Acc', alpha=0.8)
    axes[0, 0].bar(x_pos + width/2, val_accs, width, label='Val Acc', alpha=0.8)
    axes[0, 0].set_title('Training vs Validation Accuracy', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].legend()
    
    for i, (train, val) in enumerate(zip(train_accs, val_accs)):
        axes[0, 0].text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0, 0].text(i + width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Accuracy Gap Comparison
    gaps = [results_dict[model].get('accuracy_gap', 0) for model in model_names]
    axes[0, 1].bar(model_names, gaps, color=['green' if g <= 0.1 else 'orange' for g in gaps])
    axes[0, 1].set_title('Accuracy Gap (Train - Val)', fontweight='bold')
    axes[0, 1].set_ylabel('Gap')
    axes[0, 1].axhline(y=0.1, color='r', linestyle='--', label='10% threshold')
    axes[0, 1].legend()
    
    for i, gap in enumerate(gaps):
        axes[0, 1].text(i, gap + 0.005, f'{gap:.3f}', ha='center', va='bottom')
    
    # 3. F1 Score Comparison
    f1_scores = [results_dict[model].get('f1_score', 0) for model in model_names]
    axes[0, 2].bar(model_names, f1_scores, color=colors[:len(model_names)])
    axes[0, 2].set_title('Model F1 Score Comparison', fontweight='bold')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_ylim([0, 1])
    for i, f1 in enumerate(f1_scores):
        axes[0, 2].text(i, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
    
    # 4. Precision and Recall Comparison
    precisions = [results_dict[model].get('precision', 0) for model in model_names]
    recalls = [results_dict[model].get('recall', 0) for model in model_names]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, precisions, width, label='Precision', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2, recalls, width, label='Recall', alpha=0.8)
    axes[1, 0].set_title('Precision & Recall Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    
    # 5. Training Time Comparison
    training_times = [results_dict[model].get('training_time', 0) for model in model_names]
    axes[1, 1].bar(model_names, training_times, color=colors[:len(model_names)])
    axes[1, 1].set_title('Training Time Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Time (minutes)')
    max_time = max(training_times) if training_times else 1
    for i, time in enumerate(training_times):
        axes[1, 1].text(i, time + max_time*0.01, f'{time:.1f}m', ha='center', va='bottom')
    
    # 6. Overall Performance Radar Chart
    ax = axes[1, 2]
    ax.remove()  # Remove the subplot
    ax = fig.add_subplot(2, 3, 6, projection='polar')
    
    categories = ['Val Acc', 'Train Acc', 'Precision', 'Recall', 'F1-Score']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, model in enumerate(model_names):
        values = [
            results_dict[model].get('accuracy', 0),
            results_dict[model].get('train_accuracy', 0),
            results_dict[model].get('precision', 0),
            results_dict[model].get('recall', 0),
            results_dict[model].get('f1_score', 0)
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    save_path = f'{output_dir}/model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model comparison plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_model_architecture(model, model_name, output_dir, input_shape):
    """Visualize and save model architecture"""
    import io
    from contextlib import redirect_stdout
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a text representation of the model
    model_str = []
    model_str.append(f"{'='*60}")
    model_str.append(f"{model_name} Architecture")
    model_str.append(f"{'='*60}\n")
    
    # Capture model structure
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(model)
    model_str.append(buffer.getvalue())
    
    # Add parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_str.append(f"\n{'='*60}")
    model_str.append(f"Total Parameters: {total_params:,}")
    model_str.append(f"Trainable Parameters: {trainable_params:,}")
    model_str.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    model_str.append(f"{'='*60}\n")
    
    # Add layer-wise parameter count
    model_str.append("\nLayer-wise Parameter Count:")
    model_str.append("-" * 40)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                model_str.append(f"{name}: {params:,} parameters")
    
    # Save to text file
    with open(f"{output_dir}/{model_name.lower()}_architecture.txt", 'w') as f:
        f.write('\n'.join(model_str))
    
    # Create a visual representation
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.text(0.05, 0.95, '\n'.join(model_str[:50]), transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    plt.title(f'{model_name} Architecture', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name.lower()}_architecture.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Model architecture saved to {output_dir}/{model_name.lower()}_architecture.png")

# ================== PREDICTION FUNCTION ==================
def predict_model(args):
    """Predict using trained model - FIXED for new architectures"""
    # GPU/CPU detection and setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Print device information
    if device.type == 'cuda':
        print(f"🚀 Using GPU for prediction: {torch.cuda.get_device_name(0)}")
    else:
        print(f"💻 Using CPU for prediction")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_type = checkpoint.get('model_type', 'improved')
    num_classes = checkpoint.get('num_classes', len(checkpoint['classes']))
    
    print(f"Loading {model_type} model for {num_classes} classes...")
    print(f"Model trained with Train Acc: {checkpoint.get('train_acc', 0):.2%}, Val Acc: {checkpoint.get('val_acc', 0):.2%}")
    
    # Create appropriate model with correct input sizes
    if model_type == 'improved':
        input_size = checkpoint.get('input_size', 138)
        model = AdaptiveImprovedCNNLSTM(
            num_classes, 
            input_size=input_size,
            dropout_rate=0
        ).to(device)
        processor = ImprovedKeypointProcessor()
        feature_size = 138
    elif model_type == 'advanced':
        # Use stored input size or calculate it
        input_size = checkpoint.get('input_size', 220)
        model = AdaptiveAdvancedCNNLSTM(
            num_classes,
            input_size=input_size,
            dropout_rate=0
        ).to(device)
        processor = AdvancedKeypointProcessor()
        feature_size = 169  # Base features before temporal
    else:  # ensemble
        input_size_1 = checkpoint.get('input_size_1', 138)
        input_size_2 = checkpoint.get('input_size_2', 220)
        model = AdaptiveEnsembleModel(
            num_classes,
            input_size_1=input_size_1,
            input_size_2=input_size_2
        ).to(device)
        processor1 = ImprovedKeypointProcessor()
        processor2 = AdvancedKeypointProcessor()
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Process video
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if model_type == 'ensemble':
        keypoints1, keypoints2 = [], []
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
            model_complexity=1
        ) as holistic:
            
            num_frames = checkpoint['num_frames']
            
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    keypoints1.append(np.zeros(138))
                    keypoints2.append(np.zeros(169))
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                # Process for both models
                pose1 = processor1.extract_pose_keypoints(results.pose_landmarks)
                lh1 = processor1.extract_hand_keypoints(results.left_hand_landmarks)
                rh1 = processor1.extract_hand_keypoints(results.right_hand_landmarks)
                frame_features1 = np.concatenate([pose1, lh1, rh1])
                keypoints1.append(frame_features1)
                
                pose2 = processor2.extract_pose_keypoints(results.pose_landmarks)
                lh2 = processor2.extract_hand_keypoints(results.left_hand_landmarks)
                rh2 = processor2.extract_hand_keypoints(results.right_hand_landmarks)
                frame_features2 = np.concatenate([pose2, lh2, rh2])
                keypoints2.append(frame_features2)
        
        cap.release()
        
        # Process keypoints
        while len(keypoints1) < num_frames:
            if keypoints1:
                keypoints1.append(keypoints1[-1])
                keypoints2.append(keypoints2[-1])
            else:
                keypoints1.append(np.zeros(138))
                keypoints2.append(np.zeros(169))
        
        keypoints1 = np.array(keypoints1[:num_frames])
        keypoints2 = np.array(keypoints2[:num_frames])
        
        # Process keypoints2 with temporal features first, then normalize
        keypoints2 = processor2.compute_temporal_features(keypoints2)
        keypoints1 = processor1.normalize_keypoints(keypoints1)
        keypoints2 = processor2.normalize_keypoints(keypoints2)
        
        # Predict
        inputs1 = torch.tensor(keypoints1).unsqueeze(0).float().to(device)
        inputs2 = torch.tensor(keypoints2).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            ensemble_out, out1, out2 = model(inputs1, inputs2)
            probs = torch.nn.functional.softmax(ensemble_out, dim=1)
            confidence, prediction = torch.max(probs, 1)
    
    else:
        keypoints = []
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
            model_complexity=1  # Consistent complexity
        ) as holistic:
            
            num_frames = checkpoint['num_frames']
            
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    keypoints.append(np.zeros(feature_size))
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                pose = processor.extract_pose_keypoints(results.pose_landmarks)
                lh = processor.extract_hand_keypoints(results.left_hand_landmarks)
                rh = processor.extract_hand_keypoints(results.right_hand_landmarks)
                
                frame_features = np.concatenate([pose, lh, rh])
                keypoints.append(frame_features)
        
        cap.release()
        
        # Process keypoints
        while len(keypoints) < num_frames:
            if keypoints:
                keypoints.append(keypoints[-1])
            else:
                keypoints.append(np.zeros(feature_size))
        
        keypoints = np.array(keypoints[:num_frames])
        
        # Apply temporal features first for advanced, then normalize
        if model_type == 'advanced':
            keypoints = processor.compute_temporal_features(keypoints)
        keypoints = processor.normalize_keypoints(keypoints)
        
        # Predict
        inputs = torch.tensor(keypoints).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)
    
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULTS ({model_type.upper()} MODEL)")
    print(f"{'='*50}")
    print(f"Prediction: {idx_to_class[prediction.item()]}")
    print(f"Confidence: {confidence.item():.2%}")
    
    # Show top predictions
    top_k = min(5, len(checkpoint['classes']))
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    print(f"\nTop {top_k} Predictions:")
    for i in range(len(top_indices[0])):
        class_name = idx_to_class[top_indices[0][i].item()]
        prob = top_probs[0][i].item()
        print(f"  {i+1}. {class_name}: {prob:.2%}")
    
    return prediction.item(), confidence.item()

# ================== MAIN EXECUTION ==================
def check_gpu_availability():
    """Check and print GPU availability information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - CUDA Capability: {props.major}.{props.minor}")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
    else:
        print(f"✗ CUDA is not available. Running on CPU.")
        print(f"  PyTorch version: {torch.__version__}")
        
        # Check why CUDA might not be available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\n  Note: NVIDIA GPU detected but PyTorch CUDA not available.")
                print("  Consider installing PyTorch with CUDA support:")
                print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        except:
            print("\n  No NVIDIA GPU detected or nvidia-smi not available.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Check GPU availability first
    check_gpu_availability()
    
    parser = argparse.ArgumentParser(description="Enhanced Sign Language Recognition with Adaptive Multi-Class Support")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('data_dir', help="Path to dataset directory")
    train_parser.add_argument('--classes', help="Comma-separated list of specific classes to use")
    train_parser.add_argument('--num_classes', type=int, default=8, 
                            help="Number of classes to automatically select (ignored if --classes is specified)")
    train_parser.add_argument('--output_dir', default='adaptive_model_results', help="Output directory")
    train_parser.add_argument('--epochs', type=int, default=50, help="Base number of epochs")
    train_parser.add_argument('--batch_size', type=int, default=8, help="Base batch size")
    train_parser.add_argument('--num_frames', type=int, default=35, help="Frames per video")
    train_parser.add_argument('--test_size', type=float, default=0.2, help="Validation split ratio")
    train_parser.add_argument('--lr', type=float, default=0.001, help="Base learning rate")
    train_parser.add_argument('--model_type', choices=['improved', 'advanced', 'ensemble', 'all'], 
                            default='all', help="Model type to train")
    train_parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA")
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict using trained model')
    predict_parser.add_argument('model_path', help="Path to model checkpoint")
    predict_parser.add_argument('video_path', help="Path to video file for prediction")
    predict_parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Validate arguments
        if args.classes and args.num_classes != 8:
            print("Warning: --classes specified, ignoring --num_classes")
        
        if args.model_type == 'all':
            train_all_models(args)
        elif args.model_type == 'ensemble':
            train_ensemble_model(args)
        else:
            train_single_model(args, args.model_type)
            
    elif args.command == 'predict':
        predict_model(args)
    else:
        parser.print_help()