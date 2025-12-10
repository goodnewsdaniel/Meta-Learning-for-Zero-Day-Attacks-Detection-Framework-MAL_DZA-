################################################
'''   Goodnews Daniel (PhD)
        222166453@student.uj.ac.za
Department of Electrical & Electronics Engineering
Faculty of Engineering & the Built Environment
University of Johannesburg, South Africa

MAL-ZDA: Multi-level Adaptive Learning for
Zero-Day Attack Detection
Hierarchical Few-Shot Learning Framework       '''
################################################


################################################
# SECTION 1: IMPORTS AND CONSTANTS              #
################################################

# Standard libraries
import os
import json
import time
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict

# Numerical computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata

# Machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Data processing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure warnings and backend
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Non-interactive backend

# Configure plotting settings
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '.8',
    'grid.linestyle': '--'
})

# Global constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BINS = 10
BATCH_SIZE = 32
DEFAULT_N_WAY = 5
DEFAULT_K_SHOT = 1
DEFAULT_N_QUERY = 15
DEFAULT_EMBEDDING_DIM = 128

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# File paths
DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results_malzda")
RESULTS_DIR.mkdir(exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


################################################
# SECTION 2: DATA LOADING AND PREPROCESSING    #
################################################

def load_and_preprocess_real_data(
    data_path: Path,
    target_col: str = 'target'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Load and preprocess real CSV dataset with robust handling.
    Returns: X_scaled, X_unscaled, y, feature_names, kill_chain_labels
    """
    try:
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)

        print(f"Initial data shape: {df.shape}")

        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')

        # Detect target column
        possible_target_names = ['target', 'label',
                                 'Label', 'TARGET', 'LABEL', 'class', 'Class']
        target_col_name = None

        for col_name in possible_target_names:
            if col_name in df.columns:
                target_col_name = col_name
                break

        # If no target column found, assume last column is target
        if target_col_name is None:
            target_col_name = df.columns[-1]
            print(
                f"Warning: No standard target column found. Using last column: {target_col_name}")

        # Handle column names
        if not all(isinstance(col, int) for col in df.columns):
            new_columns = []
            for col in df.columns:
                if col != target_col_name:
                    words = str(col).split()
                    camel_case = '_'.join(word.capitalize() for word in words)
                    new_columns.append(camel_case)
            new_columns.append('target')
            df.columns = new_columns
        else:
            df.columns = [f'Feature_{i}' for i in range(
                df.shape[1]-1)] + ['target']

        feature_names = df.columns[:-1].tolist()
        feature_cols = df.columns[:-1]
        target_col = df.columns[-1]

        # Handle numeric features
        numeric_features = df[feature_cols].select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']).columns

        for col in numeric_features:
            # Replace infinite values with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Clip outliers
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                q1 = valid_values.quantile(0.01)
                q3 = valid_values.quantile(0.99)
                df[col] = df[col].clip(q1, q3)

            # Fill NaN with median
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

            # Convert to float
            df[col] = df[col].astype(np.float64)

        # Handle categorical features
        categorical_features = df[feature_cols].select_dtypes(
            include=['object', 'category']).columns

        label_encoders = {}
        for col in categorical_features:
            label_encoders[col] = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))

        # Handle target column - ensure it's numeric
        if df[target_col].dtype == 'object':
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(
                df[target_col].astype(str))

        # Handle class imbalance
        class_counts = df[target_col].value_counts()
        min_class_count = class_counts.min()

        if min_class_count < 2:
            print("\nWarning: Severe class imbalance detected!")
            valid_classes = class_counts[class_counts >= 2].index
            df = df[df[target_col].isin(valid_classes)]
            print(
                f"Keeping only classes with ≥2 samples: {list(valid_classes)}")

        # Extract features and labels
        X = df[feature_cols].values.astype(np.float64)
        y = df[target_col].values.astype(np.int64)

        # Store unscaled data
        X_unscaled = X.copy()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Generate synthetic kill-chain labels based on class distribution
        unique_classes = np.unique(y)
        kill_chain_labels = np.zeros(len(y), dtype=np.int64)

        for i, sample_class in enumerate(y):
            class_idx = np.where(unique_classes == sample_class)[0][0]
            phase_probs = [0.4, 0.2, 0.2, 0.1, 0.1]

            # Adjust probabilities based on class
            if class_idx == 0:
                phase_probs = [0.8, 0.1, 0.05, 0.03, 0.02]
            elif class_idx % 4 == 1:
                phase_probs = [0.0, 0.6, 0.3, 0.05, 0.05]
            elif class_idx % 4 == 2:
                phase_probs = [0.0, 0.1, 0.7, 0.15, 0.05]
            elif class_idx % 4 == 3:
                phase_probs = [0.0, 0.05, 0.1, 0.5, 0.35]

            kill_chain_labels[i] = np.random.choice(
                len(phase_probs), p=phase_probs)

        print("\nData Processing Summary:")
        print("-" * 60)
        print(f"Final shape: X={X_scaled.shape}, y={y.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Classes: {len(unique_classes)}")
        print(
            f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(
            f"Kill-chain distribution: {dict(zip(*np.unique(kill_chain_labels, return_counts=True)))}")

        return X_scaled, X_unscaled, y, feature_names, kill_chain_labels

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


class CyberSecurityDataset(Dataset):
    """Synthetic or real cybersecurity dataset with hierarchical features"""

    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        kill_chain_labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        num_classes: int = 15,
        samples_per_class: int = 1000,
        feature_dim: int = 256,
        temporal_length: int = 100,
        mode: str = 'train'
    ):
        """Initialize dataset with either real or synthetic data."""
        self.mode = mode
        self.temporal_length = temporal_length
        self.feature_dim = feature_dim
        self.kill_chain_phases = 5

        # Use real data if provided, otherwise generate synthetic
        if X is not None and y is not None:
            self._load_real_data(X, y, kill_chain_labels, feature_names)
        else:
            self._generate_synthetic_data(num_classes, samples_per_class)

    def _load_real_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kill_chain_labels: Optional[np.ndarray],
        feature_names: Optional[List[str]]
    ):
        """Load and process real data"""
        self.feature_dim = X.shape[1]
        self.num_classes = len(np.unique(y))

        self.data = []
        self.labels = []
        self.kill_chain_labels = []

        for i in range(len(X)):
            # Create hierarchical representation from flat features
            packet_features = X[i].astype(np.float32)

            # Generate flow features (temporal)
            flow_features = self._create_temporal_from_static(packet_features)

            # Generate campaign features (aggregated)
            campaign_features = np.concatenate([
                np.mean(flow_features, axis=0),
                np.std(flow_features, axis=0),
                np.max(flow_features, axis=0),
                np.min(flow_features, axis=0)
            ]).astype(np.float32)

            sample = {
                'packet': packet_features,
                'flow': flow_features,
                'campaign': campaign_features,
                'class_id': int(y[i]),
                'kill_chain': int(kill_chain_labels[i]) if kill_chain_labels is not None else 0
            }

            self.data.append(sample)
            self.labels.append(int(y[i]))
            self.kill_chain_labels.append(
                int(kill_chain_labels[i]
                    ) if kill_chain_labels is not None else 0
            )

        print(
            f"Loaded {len(self.data)} real samples with {self.num_classes} classes")

    def _create_temporal_from_static(self, features: np.ndarray) -> np.ndarray:
        """Create temporal sequence from static features"""
        flow_seq = np.zeros(
            (self.temporal_length, len(features)), dtype=np.float32)

        for t in range(self.temporal_length):
            # Add temporal variation
            noise = np.random.normal(0, 0.05, len(features))
            flow_seq[t] = features + noise

            # Add temporal dynamics
            if t > 0:
                flow_seq[t] += 0.1 * (flow_seq[t-1] - features)

        return flow_seq

    def _generate_packet_features(self, base_pattern: np.ndarray,
                                  class_id: int) -> np.ndarray:
        """Generate packet-level features with class-specific characteristics"""
        features = base_pattern.copy()

        # Safe feature dimension handling
        feature_dim = len(features)

        # Class-specific noise pattern
        class_noise = 0.1 + (class_id * 0.05)
        features += np.random.normal(0, class_noise, feature_dim)

        # Add class-specific spike patterns (safe bounds checking)
        spike_strength = 0.5 + (class_id * 0.2)
        spike_positions = np.random.choice(
            feature_dim, size=max(1, feature_dim // 5), replace=False
        )
        features[spike_positions] += spike_strength

        # Add temporal correlation pattern
        for i in range(1, min(10, feature_dim)):
            features[i] += 0.3 * features[i-1]

        # Clip to prevent extreme values
        features = np.clip(features, -5, 5)

        return features

    def _generate_flow_sequence(self, base_pattern: np.ndarray,
                                temporal_length: int, class_id: int) -> np.ndarray:
        """Generate flow-level temporal sequence"""
        sequence = np.zeros((temporal_length, len(base_pattern)))

        for t in range(temporal_length):
            # Evolving pattern over time
            time_factor = 1.0 + (0.1 * t / max(1, temporal_length - 1))
            sequence[t] = base_pattern * time_factor

            # Add class-specific evolution
            class_evolution = 0.05 * (class_id + 1)
            sequence[t] += np.random.normal(0,
                                            class_evolution, len(base_pattern))

        # Clip to prevent extreme values
        sequence = np.clip(sequence, -5, 5)

        return sequence

    def _generate_campaign_context(self, base_pattern: np.ndarray,
                                   class_id: int) -> np.ndarray:
        """Generate campaign-level context features"""
        feature_dim = len(base_pattern)

        # Create multi-scale representation
        campaign_features = np.concatenate([
            base_pattern,  # Original features
            np.roll(base_pattern, 1),  # Shifted features
            base_pattern ** 2,  # Squared features
            np.abs(np.diff(np.concatenate([[0], base_pattern])))  # Differences
        ])

        # Add class-specific context
        context_noise = 0.05 + (class_id * 0.02)
        campaign_features += np.random.normal(0,
                                              context_noise, len(campaign_features))

        # Clip to prevent extreme values
        campaign_features = np.clip(campaign_features, -5, 5)

        return campaign_features

    def _generate_synthetic_data(self, num_classes: int,
                                 samples_per_class: int) -> None:
        """Generate synthetic cyber security dataset"""
        self.data = []

        for class_id in range(num_classes):
            # Class-specific base pattern
            base_pattern = np.sin(
                np.linspace(0, 4 * np.pi, self.feature_dim) +
                (class_id * np.pi / num_classes)
            )

            for sample_id in range(samples_per_class):
                # Generate hierarchical features
                packet_features = self._generate_packet_features(
                    base_pattern, class_id)
                flow_sequence = self._generate_flow_sequence(
                    base_pattern, self.temporal_length, class_id
                )
                campaign_context = self._generate_campaign_context(
                    base_pattern, class_id
                )

                # Random kill-chain phase assignment
                kill_chain_phase = np.random.randint(0, 5)

                # Create sample
                sample = {
                    'packet': packet_features,
                    'flow': flow_sequence,
                    'campaign': campaign_context,
                    'class_id': class_id,
                    'kill_chain': kill_chain_phase
                }

                self.data.append(sample)

        print(f"✓ Generated {len(self.data)} synthetic samples "
              f"({num_classes} classes × {samples_per_class} samples)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


################################################
# SECTION 3: HIERARCHICAL ENCODER ARCHITECTURE #
################################################

class HierarchicalEncoder(nn.Module):
    """Multi-level encoder: packet, flow, and campaign levels"""

    def __init__(
        self,
        packet_dim: int = 256,
        flow_seq_len: int = 100,
        campaign_dim: int = 1024,
        embedding_dim: int = 128
    ):
        super(HierarchicalEncoder, self).__init__()

        self.packet_dim = packet_dim
        self.flow_seq_len = flow_seq_len
        self.campaign_dim = campaign_dim
        self.embedding_dim = embedding_dim

        # Packet-level encoder (1D-CNN)
        self.packet_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Flow-level encoder (Bi-LSTM)
        self.flow_encoder = nn.LSTM(
            input_size=packet_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.flow_projection = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Campaign-level encoder
        self.campaign_encoder = nn.Sequential(
            nn.Linear(campaign_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, packet_data, flow_data, campaign_data):
        """Forward pass through all hierarchical levels"""
        # Packet-level encoding
        packet_embedded = self._encode_packet(packet_data)

        # Flow-level encoding
        flow_embedded = self._encode_flow(flow_data)

        # Campaign-level encoding
        campaign_embedded = self._encode_campaign(campaign_data)

        # Normalize embeddings
        packet_embedded = F.normalize(packet_embedded, p=2, dim=-1)
        flow_embedded = F.normalize(flow_embedded, p=2, dim=-1)
        campaign_embedded = F.normalize(campaign_embedded, p=2, dim=-1)

        return packet_embedded, flow_embedded, campaign_embedded

    def _encode_packet(self, packet_data):
        """Encode packet-level features"""
        if len(packet_data.shape) == 2:
            packet_data = packet_data.unsqueeze(1)
        return self.packet_encoder(packet_data)

    def _encode_flow(self, flow_data):
        """Encode flow-level sequences"""
        lstm_out, (hidden, _) = self.flow_encoder(flow_data)
        flow_encoded = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.flow_projection(flow_encoded)

    def _encode_campaign(self, campaign_data):
        """Encode campaign-level features"""
        return self.campaign_encoder(campaign_data)


################################################
# SECTION 4: MAL-ZDA MODEL IMPLEMENTATION      #
################################################

class MALZDA(nn.Module):
    """
    Multi-level Adaptive Learning for Zero-Day Attack Detection
    Hierarchical Prototypical Network
    """

    def __init__(
        self,
        packet_dim: int = 256,
        flow_seq_len: int = 100,
        campaign_dim: int = 1024,
        embedding_dim: int = 128
    ):
        super(MALZDA, self).__init__()

        self.embedding_dim = embedding_dim
        self.hierarchical_encoder = HierarchicalEncoder(
            packet_dim, flow_seq_len, campaign_dim, embedding_dim
        )

        # Learnable distance weighting parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Temperature parameter for distance scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, support_data, query_data):
        """Forward pass for episodic training"""
        support_embeddings = self._encode_batch(support_data)
        query_embeddings = self._encode_batch(query_data)
        return support_embeddings, query_embeddings

    def _encode_batch(self, batch_data):
        """Encode a batch of samples"""
        packet_data = torch.stack([item['packet'] for item in batch_data])
        flow_data = torch.stack([item['flow'] for item in batch_data])
        campaign_data = torch.stack([item['campaign'] for item in batch_data])

        return self.hierarchical_encoder(packet_data, flow_data, campaign_data)

    def compute_prototypes(self, support_embeddings, support_labels):
        """Compute hierarchical class prototypes"""
        unique_classes = torch.unique(support_labels)
        prototypes = {}

        for class_id in unique_classes:
            class_mask = (support_labels == class_id)

            packet_proto = support_embeddings[0][class_mask].mean(dim=0)
            flow_proto = support_embeddings[1][class_mask].mean(dim=0)
            campaign_proto = support_embeddings[2][class_mask].mean(dim=0)

            prototypes[class_id.item()] = {
                'packet': packet_proto,
                'flow': flow_proto,
                'campaign': campaign_proto
            }

        return prototypes

    def compute_distances(self, query_embeddings, prototypes):
        """Compute hierarchical distances to prototypes"""
        batch_size = query_embeddings[0].shape[0]
        num_classes = len(prototypes)

        distances = torch.zeros(batch_size, num_classes,
                                device=query_embeddings[0].device)

        # Ensure weights are positive
        alpha = torch.abs(self.alpha)
        beta = torch.abs(self.beta)
        gamma = torch.abs(self.gamma)

        # Get ordered class IDs for consistent indexing
        class_ids = sorted(prototypes.keys())

        for i in range(batch_size):
            for j, class_id in enumerate(class_ids):
                proto = prototypes[class_id]

                # Euclidean distances at each level
                packet_dist = torch.norm(
                    query_embeddings[0][i] - proto['packet'])
                flow_dist = torch.norm(query_embeddings[1][i] - proto['flow'])
                campaign_dist = torch.norm(
                    query_embeddings[2][i] - proto['campaign'])

                # Weighted combination
                total_dist = (alpha * packet_dist +
                              beta * flow_dist +
                              gamma * campaign_dist)

                distances[i, j] = total_dist

        # Apply temperature scaling
        distances = distances / (torch.abs(self.temperature) + 1e-10)

        return distances

    def get_distance_weights(self):
        """Get current distance weights"""
        return {
            'alpha': torch.abs(self.alpha).item(),
            'beta': torch.abs(self.beta).item(),
            'gamma': torch.abs(self.gamma).item(),
            'temperature': torch.abs(self.temperature).item()
        }


################################################
# SECTION 5: COMPOSITIONAL TASK SAMPLING (FIXED)       #
################################################

class CompositionalTaskSampler:
    """Samples few-shot tasks with kill-chain composition"""

    def __init__(
        self,
        dataset: CyberSecurityDataset,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        include_kill_chain: bool = True
    ):
        self.dataset = dataset
        self.original_n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.include_kill_chain = include_kill_chain

        self.class_phase_indices = self._build_indices()

        # Dynamically adjust n_way based on available classes
        available_classes_count = len(self.class_phase_indices.keys())
        self.n_way = min(self.original_n_way, max(2, available_classes_count))

        if self.n_way < self.original_n_way:
            print(f"\n⚠️  Warning: Adjusted n_way from {self.original_n_way} to {self.n_way} "
                  f"(only {available_classes_count} classes available)")

    def _build_indices(self):
        """Build index structure for efficient sampling"""
        class_phase_indices = defaultdict(lambda: defaultdict(list))

        for idx, sample in enumerate(self.dataset):
            class_id = sample['class_id']
            kill_chain = sample['kill_chain']
            class_phase_indices[class_id][kill_chain].append(idx)

        return class_phase_indices

    def sample_task(self):
        """Sample a few-shot learning task"""
        if self.include_kill_chain:
            return self._sample_compositional_task()
        else:
            return self._sample_standard_task()

    def _sample_compositional_task(self):
        """Sample task with kill-chain phase diversity"""
        available_classes = list(self.class_phase_indices.keys())

        if len(available_classes) < self.n_way:
            # Fallback to standard sampling with available classes
            return self._sample_standard_task()

        selected_classes = np.random.choice(
            available_classes, self.n_way, replace=False
        )

        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []

        for class_idx, class_id in enumerate(selected_classes):
            available_phases = list(self.class_phase_indices[class_id].keys())

            # Sample support from different phases
            support_class_indices = []
            for shot_idx in range(self.k_shot):
                phase = available_phases[shot_idx % len(available_phases)]
                phase_indices = self.class_phase_indices[class_id][phase]

                if len(phase_indices) > 0:
                    support_idx = np.random.choice(phase_indices)
                    support_class_indices.append(support_idx)

            support_indices.extend(support_class_indices[:self.k_shot])
            support_labels.extend(
                [class_idx] * len(support_class_indices[:self.k_shot]))

            # Sample query samples
            all_query_indices = []
            for phase in available_phases:
                all_query_indices.extend(
                    self.class_phase_indices[class_id][phase])

            query_candidates = [idx for idx in all_query_indices
                                if idx not in support_class_indices]

            if len(query_candidates) >= self.n_query:
                query_class_indices = np.random.choice(
                    query_candidates, self.n_query, replace=False
                )
            else:
                # Use all available and sample with replacement if needed
                if len(query_candidates) > 0:
                    query_class_indices = np.random.choice(
                        query_candidates,
                        min(self.n_query, len(query_candidates)),
                        replace=False
                    )
                else:
                    # Fallback: sample from support indices (data augmentation)
                    query_class_indices = np.random.choice(
                        support_class_indices,
                        min(self.n_query, len(support_class_indices)),
                        replace=True
                    )

            query_indices.extend(query_class_indices)
            query_labels.extend([class_idx] * len(query_class_indices))

        support_set = [self.dataset[idx] for idx in support_indices]
        query_set = [self.dataset[idx] for idx in query_indices]

        return support_set, query_set, support_labels, query_labels

    def _sample_standard_task(self):
        """Sample standard few-shot task"""
        available_classes = list(self.class_phase_indices.keys())

        # Adaptive n_way selection
        effective_n_way = min(self.n_way, len(available_classes))

        if len(available_classes) < effective_n_way:
            effective_n_way = max(2, len(available_classes))

        selected_classes = np.random.choice(
            available_classes, effective_n_way, replace=False
        )

        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []

        for class_idx, class_id in enumerate(selected_classes):
            all_indices = []
            for phase in self.class_phase_indices[class_id]:
                all_indices.extend(self.class_phase_indices[class_id][phase])

            all_indices = list(set(all_indices))  # Remove duplicates

            if len(all_indices) >= self.k_shot + self.n_query:
                selected = np.random.choice(
                    all_indices, self.k_shot + self.n_query, replace=False
                )
                support_indices.extend(selected[:self.k_shot])
                query_indices.extend(selected[self.k_shot:])
            elif len(all_indices) >= self.k_shot:
                # Sample with replacement for query if needed
                support_indices.extend(np.random.choice(
                    all_indices, self.k_shot, replace=False
                ))
                remaining_for_query = self.n_query
                if len(all_indices) > self.k_shot:
                    # Get non-support indices
                    query_candidate_indices = [
                        idx for idx in all_indices
                        if idx not in support_indices[-self.k_shot:]
                    ]
                    if query_candidate_indices:
                        take = min(remaining_for_query, len(
                            query_candidate_indices))
                        query_indices.extend(np.random.choice(
                            query_candidate_indices, take, replace=False
                        ))
                        remaining_for_query -= take

                if remaining_for_query > 0:
                    query_indices.extend(np.random.choice(
                        all_indices, remaining_for_query, replace=True
                    ))
            else:
                # Very few samples: use all with replacement
                support_indices.extend(np.random.choice(
                    all_indices, self.k_shot, replace=True
                ))
                query_indices.extend(np.random.choice(
                    all_indices, self.n_query, replace=True
                ))

            support_labels.extend([class_idx] * self.k_shot)
            query_labels.extend([class_idx] * len([
                idx for idx in query_indices
                if query_indices.index(idx) >= len(query_indices) -
                (self.n_query if class_idx == len(selected_classes) - 1 else 0)
            ]) if class_idx == len(selected_classes) - 1 else [class_idx] * self.n_query)

        # Ensure query_labels has correct length
        if len(query_labels) < len(query_indices):
            query_labels.extend([selected_classes[-1]] *
                                (len(query_indices) - len(query_labels)))
        query_labels = query_labels[:len(query_indices)]

        support_set = [self.dataset[idx] for idx in support_indices]
        query_set = [self.dataset[idx] for idx in query_indices]

        return support_set, query_set, support_labels, query_labels


# SECTION 6: TRAINING AND EVALUATION           #
################################################

class MALZDATrainer:
    """Training and evaluation framework"""

    def __init__(
        self,
        model: MALZDA,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )

        self.training_history = {
            'loss': [],
            'accuracy': [],
            'distance_weights': []
        }

    def train_episode(self, support_set, query_set, support_labels, query_labels):
        """Train on single episode"""
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        for item in support_set:
            item['packet'] = torch.tensor(
                item['packet'], dtype=torch.float32).to(self.device)
            item['flow'] = torch.tensor(
                item['flow'], dtype=torch.float32).to(self.device)
            item['campaign'] = torch.tensor(
                item['campaign'], dtype=torch.float32).to(self.device)

        for item in query_set:
            item['packet'] = torch.tensor(
                item['packet'], dtype=torch.float32).to(self.device)
            item['flow'] = torch.tensor(
                item['flow'], dtype=torch.float32).to(self.device)
            item['campaign'] = torch.tensor(
                item['campaign'], dtype=torch.float32).to(self.device)

        support_labels = torch.tensor(
            support_labels, dtype=torch.long).to(self.device)
        query_labels = torch.tensor(
            query_labels, dtype=torch.long).to(self.device)

        # Forward pass
        support_embeddings, query_embeddings = self.model(
            support_set, query_set)

        # Compute prototypes and distances
        prototypes = self.model.compute_prototypes(
            support_embeddings, support_labels)
        distances = self.model.compute_distances(query_embeddings, prototypes)

        # Compute loss
        log_probs = F.log_softmax(-distances, dim=1)
        loss = F.nll_loss(log_probs, query_labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute accuracy
        predictions = torch.argmin(distances, dim=1)
        accuracy = (predictions == query_labels).float().mean().item()

        # Store history
        self.training_history['loss'].append(loss.item())
        self.training_history['accuracy'].append(accuracy)
        self.training_history['distance_weights'].append(
            self.model.get_distance_weights()
        )

        return loss.item(), accuracy

    def evaluate_episode(self, support_set, query_set, support_labels, query_labels):
        """Evaluate on single episode"""
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            for item in support_set:
                item['packet'] = torch.tensor(
                    item['packet'], dtype=torch.float32).to(self.device)
                item['flow'] = torch.tensor(
                    item['flow'], dtype=torch.float32).to(self.device)
                item['campaign'] = torch.tensor(
                    item['campaign'], dtype=torch.float32).to(self.device)

            for item in query_set:
                item['packet'] = torch.tensor(
                    item['packet'], dtype=torch.float32).to(self.device)
                item['flow'] = torch.tensor(
                    item['flow'], dtype=torch.float32).to(self.device)
                item['campaign'] = torch.tensor(
                    item['campaign'], dtype=torch.float32).to(self.device)

            support_labels = torch.tensor(
                support_labels, dtype=torch.long).to(self.device)
            query_labels = torch.tensor(
                query_labels, dtype=torch.long).to(self.device)

            # Forward pass
            support_embeddings, query_embeddings = self.model(
                support_set, query_set)

            # Compute prototypes and distances
            prototypes = self.model.compute_prototypes(
                support_embeddings, support_labels)
            distances = self.model.compute_distances(
                query_embeddings, prototypes)

            # Compute metrics
            log_probs = F.log_softmax(-distances, dim=1)
            loss = F.nll_loss(log_probs, query_labels)

            predictions = torch.argmin(distances, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()

            # Detailed metrics
            query_labels_np = query_labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            f1 = f1_score(query_labels_np, predictions_np,
                          average='macro', zero_division=0)
            precision = precision_score(
                query_labels_np, predictions_np, average='macro', zero_division=0)
            recall = recall_score(
                query_labels_np, predictions_np, average='macro', zero_division=0)

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': predictions_np,
            'labels': query_labels_np,
            'distance_weights': self.model.get_distance_weights()
        }

    def save_model(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {path}")


################################################
# SECTION 7: EXPERIMENTAL FRAMEWORK (FIXED)            #
################################################

def run_experiment(
    dataset_train: CyberSecurityDataset,
    dataset_test: CyberSecurityDataset,
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    num_episodes: int = 1000,
    eval_episodes: int = 200,
    use_compositional: bool = True,
    experiment_name: str = "malzda_experiment"
):
    """Run complete MAL-ZDA experiment"""

    print("\n" + "="*80)
    print(f"MAL-ZDA Experiment: {experiment_name}")
    print(f"Configuration: {n_way}-way {k_shot}-shot with {n_query} queries")
    print(f"Compositional Sampling: {use_compositional}")
    print("="*80)

    # Create task samplers with automatic n_way adjustment
    train_sampler = CompositionalTaskSampler(
        dataset_train, n_way=n_way, k_shot=k_shot,
        n_query=n_query, include_kill_chain=use_compositional
    )
    test_sampler = CompositionalTaskSampler(
        dataset_test, n_way=n_way, k_shot=k_shot,
        n_query=n_query, include_kill_chain=use_compositional
    )

    # Determine dimensions from dataset
    sample = dataset_train[0]
    packet_dim = len(sample['packet'])
    flow_seq_len = len(sample['flow'])
    campaign_dim = len(sample['campaign'])

    print(f"\nData dimensions:")
    print(f"  Packet: {packet_dim}")
    print(f"  Flow: {flow_seq_len} x {packet_dim}")
    print(f"  Campaign: {campaign_dim}")
    print(
        f"  Available train classes: {len(train_sampler.class_phase_indices)}")
    print(f"  Available test classes: {len(test_sampler.class_phase_indices)}")

    # Initialize model
    model = MALZDA(
        packet_dim=packet_dim,
        flow_seq_len=flow_seq_len,
        campaign_dim=campaign_dim,
        embedding_dim=DEFAULT_EMBEDDING_DIM
    )

    trainer = MALZDATrainer(model, learning_rate=0.001, device=DEVICE)

    # Training loop
    print(f"\nTraining for {num_episodes} episodes...")
    train_losses = []
    train_accuracies = []
    successful_episodes = 0

    for episode in tqdm(range(num_episodes), desc="Training"):
        try:
            support_set, query_set, support_labels, query_labels = train_sampler.sample_task()

            loss, accuracy = trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )

            train_losses.append(loss)
            train_accuracies.append(accuracy)
            successful_episodes += 1

            if (episode + 1) % 100 == 0:
                avg_loss = np.mean(
                    train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses)
                avg_acc = np.mean(
                    train_accuracies[-100:]) if len(train_accuracies) >= 100 else np.mean(train_accuracies)
                print(
                    f"\nEpisode {episode+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

                # Print distance weights
                weights = trainer.model.get_distance_weights()
                print(
                    f"  Weights: α={weights['alpha']:.3f}, β={weights['beta']:.3f}, γ={weights['gamma']:.3f}")

            trainer.scheduler.step()

        except Exception as e:
            # Silent fail for this episode
            continue

    print(
        f"\n✓ Completed {successful_episodes}/{num_episodes} training episodes")

    # Evaluation
    print(f"\nEvaluating on {eval_episodes} episodes...")
    eval_results = {
        'losses': [],
        'accuracies': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'all_predictions': [],
        'all_labels': []
    }

    successful_eval_episodes = 0

    for episode in tqdm(range(eval_episodes), desc="Evaluating"):
        try:
            support_set, query_set, support_labels, query_labels = test_sampler.sample_task()

            results = trainer.evaluate_episode(
                support_set, query_set, support_labels, query_labels
            )

            eval_results['losses'].append(results['loss'])
            eval_results['accuracies'].append(results['accuracy'])
            eval_results['f1_scores'].append(results['f1'])
            eval_results['precisions'].append(results['precision'])
            eval_results['recalls'].append(results['recall'])
            eval_results['all_predictions'].extend(results['predictions'])
            eval_results['all_labels'].extend(results['labels'])
            successful_eval_episodes += 1

        except Exception as e:
            # Silent fail
            continue

    print(
        f"✓ Completed {successful_eval_episodes}/{eval_episodes} evaluation episodes")

    # Print results with safety checks
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    if len(eval_results['accuracies']) > 0:
        acc_mean = np.mean(eval_results['accuracies'])
        acc_std = np.std(eval_results['accuracies'])
        f1_mean = np.mean(eval_results['f1_scores'])
        f1_std = np.std(eval_results['f1_scores'])
        prec_mean = np.mean(eval_results['precisions'])
        prec_std = np.std(eval_results['precisions'])
        rec_mean = np.mean(eval_results['recalls'])
        rec_std = np.std(eval_results['recalls'])

        print(f"Test Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"Test F1 Score:  {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"Test Precision: {prec_mean:.4f} ± {prec_std:.4f}")
        print(f"Test Recall:    {rec_mean:.4f} ± {rec_std:.4f}")
    else:
        print("⚠️  No successful evaluation episodes!")
        acc_mean = 0.0
        f1_mean = 0.0
        prec_mean = 0.0
        rec_mean = 0.0

    print("="*80)

    # Save model and results
    model_path = RESULTS_DIR / f"{experiment_name}_model.pt"
    trainer.save_model(model_path)

    results_dict = {
        'config': {
            'n_way': train_sampler.n_way,
            'k_shot': k_shot,
            'n_query': n_query,
            'num_episodes': num_episodes,
            'eval_episodes': eval_episodes,
            'use_compositional': use_compositional,
            'successful_train_episodes': successful_episodes,
            'successful_eval_episodes': successful_eval_episodes
        },
        'training': {
            'losses': train_losses,
            'accuracies': train_accuracies
        },
        'evaluation': eval_results
    }

    results_path = RESULTS_DIR / f"{experiment_name}_results.json"

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else 0.0
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results_dict)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return model, eval_results, train_losses, train_accuracies


# VISUALIZATION FIXES FOR NaN HANDLING          #
################################################

def _save_individual_scaling_all_metrics(scaling_results: Dict, save_name: str):
    """Save individual scaling all metrics chart with NaN safety"""
    fig, ax = plt.subplots(figsize=(13, 7))

    k_shots = list(scaling_results.keys())
    accuracies = [scaling_results[k]['accuracy'] for k in k_shots]
    f1_scores = [scaling_results[k]['f1'] for k in k_shots]
    precisions = [scaling_results[k]['precision'] for k in k_shots]
    recalls = [scaling_results[k]['recall'] for k in k_shots]

    # Filter out NaN values
    valid_mask = ~(np.isnan(accuracies) | np.isnan(f1_scores) |
                   np.isnan(precisions) | np.isnan(recalls))

    if not np.any(valid_mask):
        print(f"⚠️  Warning: All metrics are NaN for {save_name}")
        plt.close(fig)
        return

    k_shots_valid = [k for k, v in zip(k_shots, valid_mask) if v]
    accuracies_valid = [acc for acc, v in zip(accuracies, valid_mask) if v]
    f1_scores_valid = [f1 for f1, v in zip(f1_scores, valid_mask) if v]
    precisions_valid = [prec for prec, v in zip(precisions, valid_mask) if v]
    recalls_valid = [rec for rec, v in zip(recalls, valid_mask) if v]

    ax.plot(k_shots_valid, accuracies_valid, marker='o', markersize=10,
            linewidth=2.5, label='Accuracy', color='blue')
    ax.plot(k_shots_valid, f1_scores_valid, marker='s', markersize=10,
            linewidth=2.5, label='F1 Score', color='green')
    ax.plot(k_shots_valid, precisions_valid, marker='^', markersize=10,
            linewidth=2.5, label='Precision', color='red')
    ax.plot(k_shots_valid, recalls_valid, marker='v', markersize=10,
            linewidth=2.5, label='Recall', color='purple')

    ax.set_xlabel('Number of Support Examples (k-shot)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Metrics vs Support Set Size',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_xticks(k_shots_valid)

    # Safe ylim setting
    all_valid_values = accuracies_valid + \
        f1_scores_valid + precisions_valid + recalls_valid
    if all_valid_values:
        y_min = max(0, min(all_valid_values) - 0.1)
        y_max = min(1.0, max(all_valid_values) + 0.1)
        ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual scaling all metrics saved to {save_name}_all_metrics.png")


def visualize_training_results(
    train_losses: List[float],
    train_accuracies: List[float],
    eval_results: Dict,
    save_prefix: str = "malzda"
):
    """Create comprehensive visualization of results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Training loss
    axes[0, 0].plot(train_losses, alpha=0.3,
                    label='Episode Loss', color='blue')
    if len(train_losses) > 50:
        axes[0, 0].plot(
            pd.Series(train_losses).rolling(50).mean(),
            label='Moving Average (50)', linewidth=2, color='darkblue'
        )
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Training accuracy
    axes[0, 1].plot(train_accuracies, alpha=0.3,
                    label='Episode Accuracy', color='green')
    if len(train_accuracies) > 50:
        axes[0, 1].plot(
            pd.Series(train_accuracies).rolling(50).mean(),
            label='Moving Average (50)', linewidth=2, color='darkgreen'
        )
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Evaluation metrics boxplot
    metrics_data = {
        'Accuracy': eval_results['accuracies'],
        'F1 Score': eval_results['f1_scores'],
        'Precision': eval_results['precisions'],
        'Recall': eval_results['recalls']
    }

    bp = axes[0, 2].boxplot(
        [metrics_data[k] for k in metrics_data.keys()],
        labels=list(metrics_data.keys()),
        patch_artist=True
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Evaluation Metrics Distribution')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].tick_params(axis='x', rotation=15)

    # 4. Accuracy histogram
    axes[1, 0].hist(eval_results['accuracies'], bins=20,
                    edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(
        np.mean(eval_results['accuracies']),
        color='red', linestyle='--', linewidth=2,
        label=f"Mean: {np.mean(eval_results['accuracies']):.3f}"
    )
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Test Accuracy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Confusion matrix
    if len(eval_results['all_predictions']) > 0:
        cm = confusion_matrix(
            eval_results['all_labels'], eval_results['all_predictions'])
        im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_title('Confusion Matrix')
        plt.colorbar(im, ax=axes[1, 1])

    # 6. Metrics summary table
    summary_data = [
        ['Accuracy', f"{np.mean(eval_results['accuracies']):.4f}",
         f"±{np.std(eval_results['accuracies']):.4f}"],
        ['F1 Score', f"{np.mean(eval_results['f1_scores']):.4f}",
         f"±{np.std(eval_results['f1_scores']):.4f}"],
        ['Precision', f"{np.mean(eval_results['precisions']):.4f}",
         f"±{np.std(eval_results['precisions']):.4f}"],
        ['Recall', f"{np.mean(eval_results['recalls']):.4f}",
         f"±{np.std(eval_results['recalls']):.4f}"]
    ]

    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    table = axes[1, 2].table(
        cellText=summary_data,
        colLabels=['Metric', 'Mean', 'Std Dev'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined visualization saved to {save_prefix}_results.png")

    # Now create individual visualizations
    _save_individual_training_loss(train_losses, save_prefix)
    _save_individual_training_accuracy(train_accuracies, save_prefix)
    _save_individual_metrics_boxplot(eval_results, save_prefix)
    _save_individual_accuracy_histogram(eval_results, save_prefix)
    _save_individual_confusion_matrix(eval_results, save_prefix)
    _save_individual_metrics_table(eval_results, save_prefix)


def _save_individual_training_loss(train_losses: List[float], save_prefix: str):
    """Save individual training loss visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train_losses, alpha=0.3, label='Episode Loss', color='blue')
    if len(train_losses) > 50:
        ax.plot(
            pd.Series(train_losses).rolling(50).mean(),
            label='Moving Average (50)', linewidth=2, color='darkblue'
        )
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual training loss saved to {save_prefix}_training_loss.png")


def _save_individual_training_accuracy(train_accuracies: List[float], save_prefix: str):
    """Save individual training accuracy visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train_accuracies, alpha=0.3,
            label='Episode Accuracy', color='green')
    if len(train_accuracies) > 50:
        ax.plot(
            pd.Series(train_accuracies).rolling(50).mean(),
            label='Moving Average (50)', linewidth=2, color='darkgreen'
        )
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Accuracy Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual training accuracy saved to {save_prefix}_training_accuracy.png")


def _save_individual_metrics_boxplot(eval_results: Dict, save_prefix: str):
    """Save individual metrics boxplot visualization"""
    fig, ax = plt.subplots(figsize=(10, 7))

    metrics_data = {
        'Accuracy': eval_results['accuracies'],
        'F1 Score': eval_results['f1_scores'],
        'Precision': eval_results['precisions'],
        'Recall': eval_results['recalls']
    }

    bp = ax.boxplot(
        [metrics_data[k] for k in metrics_data.keys()],
        labels=list(metrics_data.keys()),
        patch_artist=True
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Evaluation Metrics Distribution',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_metrics_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual metrics boxplot saved to {save_prefix}_metrics_boxplot.png")


def _save_individual_accuracy_histogram(eval_results: Dict, save_prefix: str):
    """Save individual accuracy histogram visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(eval_results['accuracies'], bins=20,
            edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(
        np.mean(eval_results['accuracies']),
        color='red', linestyle='--', linewidth=2,
        label=f"Mean: {np.mean(eval_results['accuracies']):.3f}"
    )
    ax.axvline(
        np.median(eval_results['accuracies']),
        color='green', linestyle='--', linewidth=2,
        label=f"Median: {np.median(eval_results['accuracies']):.3f}"
    )
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Test Accuracy Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_accuracy_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual accuracy histogram saved to {save_prefix}_accuracy_histogram.png")


def _save_individual_confusion_matrix(eval_results: Dict, save_prefix: str):
    """Save individual confusion matrix visualization"""
    if len(eval_results['all_predictions']) == 0:
        print(f"Skipping confusion matrix for {save_prefix} (no predictions)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    cm = confusion_matrix(
        eval_results['all_labels'], eval_results['all_predictions'])

    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual confusion matrix saved to {save_prefix}_confusion_matrix.png")


def _save_individual_metrics_table(eval_results: Dict, save_prefix: str):
    """Save individual metrics summary table visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_data = [
        ['Accuracy', f"{np.mean(eval_results['accuracies']):.4f}",
         f"±{np.std(eval_results['accuracies']):.4f}"],
        ['F1 Score', f"{np.mean(eval_results['f1_scores']):.4f}",
         f"±{np.std(eval_results['f1_scores']):.4f}"],
        ['Precision', f"{np.mean(eval_results['precisions']):.4f}",
         f"±{np.std(eval_results['precisions']):.4f}"],
        ['Recall', f"{np.mean(eval_results['recalls']):.4f}",
         f"±{np.std(eval_results['recalls']):.4f}"]
    ]

    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=summary_data,
        colLabels=['Metric', 'Mean', 'Std Dev'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_prefix}_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual metrics table saved to {save_prefix}_metrics_table.png")


def create_comparison_visualization(results_comp: Dict, results_std: Dict, save_name: str = "comparison"):
    """Compare compositional vs standard approaches"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Accuracy comparison
    data_acc = [results_comp['accuracies'], results_std['accuracies']]
    bp1 = axes[0, 0].boxplot(
        data_acc, labels=['Compositional', 'Standard'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. F1 Score comparison
    data_f1 = [results_comp['f1_scores'], results_std['f1_scores']]
    bp2 = axes[0, 1].boxplot(
        data_f1, labels=['Compositional', 'Standard'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Precision-Recall scatter
    axes[1, 0].scatter(
        results_comp['precisions'], results_comp['recalls'],
        alpha=0.5, label='Compositional', s=30, color='blue'
    )
    axes[1, 0].scatter(
        results_std['precisions'], results_std['recalls'],
        alpha=0.5, label='Standard', s=30, color='red'
    )
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Precision-Recall Scatter')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Mean metrics bar chart
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    comp_means = [
        np.mean(results_comp['accuracies']),
        np.mean(results_comp['f1_scores']),
        np.mean(results_comp['precisions']),
        np.mean(results_comp['recalls'])
    ]
    std_means = [
        np.mean(results_std['accuracies']),
        np.mean(results_std['f1_scores']),
        np.mean(results_std['precisions']),
        np.mean(results_std['recalls'])
    ]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = axes[1, 1].bar(x - width/2, comp_means, width,
                           label='Compositional', color='lightblue', edgecolor='black', linewidth=1.5)
    bars2 = axes[1, 1].bar(x + width/2, std_means, width, label='Standard',
                           color='lightcoral', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=10)

    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison visualization saved to {save_name}.png")

    # Create individual comparison charts
    _save_individual_accuracy_comparison(results_comp, results_std, save_name)
    _save_individual_f1_comparison(results_comp, results_std, save_name)
    _save_individual_precision_recall_scatter(
        results_comp, results_std, save_name)
    _save_individual_metrics_bar_chart(results_comp, results_std, save_name)


def _save_individual_accuracy_comparison(results_comp: Dict, results_std: Dict, save_name: str):
    """Save individual accuracy comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_acc = [results_comp['accuracies'], results_std['accuracies']]
    bp = ax.boxplot(
        data_acc, labels=['Compositional', 'Standard'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][1].set_facecolor('lightcoral')
    bp['boxes'][1].set_edgecolor('black')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison: Compositional vs Standard',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual accuracy comparison saved to {save_name}_accuracy_comparison.png")


def _save_individual_f1_comparison(results_comp: Dict, results_std: Dict, save_name: str):
    """Save individual F1 score comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_f1 = [results_comp['f1_scores'], results_std['f1_scores']]
    bp = ax.boxplot(
        data_f1, labels=['Compositional', 'Standard'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][1].set_facecolor('lightcoral')
    bp['boxes'][1].set_edgecolor('black')

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Comparison: Compositional vs Standard',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual F1 comparison saved to {save_name}_f1_comparison.png")


def _save_individual_precision_recall_scatter(results_comp: Dict, results_std: Dict, save_name: str):
    """Save individual precision-recall scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        results_comp['precisions'], results_comp['recalls'],
        alpha=0.6, label='Compositional', s=80, color='blue', edgecolor='black'
    )
    ax.scatter(
        results_std['precisions'], results_std['recalls'],
        alpha=0.6, label='Standard', s=80, color='red', edgecolor='black'
    )
    ax.set_xlabel('Precision', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Precision-Recall Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_precision_recall_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual precision-recall scatter saved to {save_name}_precision_recall_scatter.png")


def _save_individual_metrics_bar_chart(results_comp: Dict, results_std: Dict, save_name: str):
    """Save individual metrics bar chart comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    comp_means = [
        np.mean(results_comp['accuracies']),
        np.mean(results_comp['f1_scores']),
        np.mean(results_comp['precisions']),
        np.mean(results_comp['recalls'])
    ]
    std_means = [
        np.mean(results_std['accuracies']),
        np.mean(results_std['f1_scores']),
        np.mean(results_std['precisions']),
        np.mean(results_std['recalls'])
    ]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, comp_means, width,
                   label='Compositional', color='lightblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, std_means, width, label='Standard',
                   color='lightcoral', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_metrics_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual metrics bar chart saved to {save_name}_metrics_bar_chart.png")


def visualize_ablation_results(ablation_results: Dict, save_name: str = "ablation"):
    """Visualize ablation study results"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    configs = list(ablation_results.keys())
    accuracies = [ablation_results[c]['accuracy'] for c in configs]
    acc_stds = [ablation_results[c]['accuracy_std'] for c in configs]
    f1_scores = [ablation_results[c]['f1'] for c in configs]
    f1_stds = [ablation_results[c]['f1_std'] for c in configs]

    x = np.arange(len(configs))

    # Accuracy
    axes[0].bar(x, accuracies, yerr=acc_stds, capsize=5,
                color='skyblue', edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Ablation Study: Accuracy by Configuration',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, max(accuracies) * 1.2]
                     if max(accuracies) > 0 else [0, 1])

    # F1 Score
    axes[1].bar(x, f1_scores, yerr=f1_stds, capsize=5,
                color='lightcoral', edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Ablation Study: F1 Score by Configuration',
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, max(f1_scores) * 1.2]
                     if max(f1_scores) > 0 else [0, 1])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Ablation visualization saved to {save_name}.png")

    # Create individual visualizations
    _save_individual_ablation_accuracy(ablation_results, save_name)
    _save_individual_ablation_f1(ablation_results, save_name)


def _save_individual_ablation_accuracy(ablation_results: Dict, save_name: str):
    """Save individual ablation accuracy chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = list(ablation_results.keys())
    accuracies = [ablation_results[c]['accuracy'] for c in configs]
    acc_stds = [ablation_results[c]['accuracy_std'] for c in configs]

    x = np.arange(len(configs))
    bars = ax.bar(x, accuracies, yerr=acc_stds, capsize=8,
                  color='skyblue', edgecolor='black', alpha=0.8, linewidth=1.5)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2., acc,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Accuracy by Configuration',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(accuracies) * 1.2] if max(accuracies) > 0 else [0, 1])

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual ablation accuracy saved to {save_name}_accuracy.png")


def _save_individual_ablation_f1(ablation_results: Dict, save_name: str):
    """Save individual ablation F1 score chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = list(ablation_results.keys())
    f1_scores = [ablation_results[c]['f1'] for c in configs]
    f1_stds = [ablation_results[c]['f1_std'] for c in configs]

    x = np.arange(len(configs))
    bars = ax.bar(x, f1_scores, yerr=f1_stds, capsize=8,
                  color='lightcoral', edgecolor='black', alpha=0.8, linewidth=1.5)

    # Add value labels
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax.text(bar.get_x() + bar.get_width()/2., f1,
                f'{f1:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Ablation Study: F1 Score by Configuration',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(f1_scores) * 1.2] if max(f1_scores) > 0 else [0, 1])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{save_name}_f1.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual ablation F1 saved to {save_name}_f1.png")


def visualize_scaling_results(scaling_results: Dict, save_name: str = "scaling"):
    """Visualize few-shot scaling experiment results"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    k_shots = list(scaling_results.keys())
    accuracies = [scaling_results[k]['accuracy'] for k in k_shots]
    acc_stds = [scaling_results[k]['accuracy_std'] for k in k_shots]
    f1_scores = [scaling_results[k]['f1'] for k in k_shots]
    precisions = [scaling_results[k]['precision'] for k in k_shots]
    recalls = [scaling_results[k]['recall'] for k in k_shots]

    # Accuracy vs k-shot
    axes[0].errorbar(
        k_shots, accuracies, yerr=acc_stds,
        marker='o', markersize=10, linewidth=2, capsize=5,
        color='blue', label='Accuracy'
    )
    axes[0].set_xlabel('Number of Support Examples (k-shot)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Performance vs Support Set Size',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xticks(k_shots)

    # All metrics comparison
    axes[1].plot(k_shots, accuracies, marker='o',
                 linewidth=2, label='Accuracy')
    axes[1].plot(k_shots, f1_scores, marker='s', linewidth=2, label='F1 Score')
    axes[1].plot(k_shots, precisions, marker='^',
                 linewidth=2, label='Precision')
    axes[1].plot(k_shots, recalls, marker='v', linewidth=2, label='Recall')
    axes[1].set_xlabel('Number of Support Examples (k-shot)', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('All Metrics vs Support Set Size',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xticks(k_shots)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scaling visualization saved to {save_name}.png")

    # Create individual visualizations
    _save_individual_scaling_accuracy(scaling_results, save_name)
    _save_individual_scaling_all_metrics(scaling_results, save_name)


def _save_individual_scaling_accuracy(scaling_results: Dict, save_name: str):
    """Save individual scaling accuracy chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    k_shots = list(scaling_results.keys())
    accuracies = [scaling_results[k]['accuracy'] for k in k_shots]
    acc_stds = [scaling_results[k]['accuracy_std'] for k in k_shots]

    ax.errorbar(
        k_shots, accuracies, yerr=acc_stds,
        marker='o', markersize=12, linewidth=2.5, capsize=8,
        color='blue', ecolor='darkblue', label='Accuracy', elinewidth=2
    )
    ax.set_xlabel('Number of Support Examples (k-shot)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy vs Support Set Size (k-shot)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(k_shots)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_accuracy_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual scaling accuracy saved to {save_name}_accuracy_scaling.png")


def _save_individual_scaling_all_metrics(scaling_results: Dict, save_name: str):
    """Save individual scaling all metrics chart with NaN safety"""
    fig, ax = plt.subplots(figsize=(13, 7))

    k_shots = list(scaling_results.keys())
    accuracies = [scaling_results[k]['accuracy'] for k in k_shots]
    f1_scores = [scaling_results[k]['f1'] for k in k_shots]
    precisions = [scaling_results[k]['precision'] for k in k_shots]
    recalls = [scaling_results[k]['recall'] for k in k_shots]

    # Filter out NaN values
    valid_mask = ~(np.isnan(accuracies) | np.isnan(f1_scores) |
                   np.isnan(precisions) | np.isnan(recalls))

    if not np.any(valid_mask):
        print(f"⚠️  Warning: All metrics are NaN for {save_name}")
        plt.close(fig)
        return

    k_shots_valid = [k for k, v in zip(k_shots, valid_mask) if v]
    accuracies_valid = [acc for acc, v in zip(accuracies, valid_mask) if v]
    f1_scores_valid = [f1 for f1, v in zip(f1_scores, valid_mask) if v]
    precisions_valid = [prec for prec, v in zip(precisions, valid_mask) if v]
    recalls_valid = [rec for rec, v in zip(recalls, valid_mask) if v]

    ax.plot(k_shots_valid, accuracies_valid, marker='o', markersize=10,
            linewidth=2.5, label='Accuracy', color='blue')
    ax.plot(k_shots_valid, f1_scores_valid, marker='s', markersize=10,
            linewidth=2.5, label='F1 Score', color='green')
    ax.plot(k_shots_valid, precisions_valid, marker='^', markersize=10,
            linewidth=2.5, label='Precision', color='red')
    ax.plot(k_shots_valid, recalls_valid, marker='v', markersize=10,
            linewidth=2.5, label='Recall', color='purple')

    ax.set_xlabel('Number of Support Examples (k-shot)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Metrics vs Support Set Size',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_xticks(k_shots_valid)

    # Safe ylim setting
    all_valid_values = accuracies_valid + \
        f1_scores_valid + precisions_valid + recalls_valid
    if all_valid_values:
        y_min = max(0, min(all_valid_values) - 0.1)
        y_max = min(1.0, max(all_valid_values) + 0.1)
        ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f'{save_name}_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Individual scaling all metrics saved to {save_name}_all_metrics.png")


def run_ablation_study(
    dataset_train: CyberSecurityDataset,
    dataset_test: CyberSecurityDataset,
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    num_episodes: int = 500,
    eval_episodes: int = 100
):
    """Run ablation studies on hierarchical components"""

    print("\n" + "="*80)
    print("ABLATION STUDY: Impact of Hierarchical Levels")
    print("="*80)

    configurations = [
        {'name': 'Full Model', 'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0},
        {'name': 'Packet Only', 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
        {'name': 'Flow Only', 'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0},
        {'name': 'Campaign Only', 'alpha': 0.0, 'beta': 0.0, 'gamma': 1.0},
        {'name': 'Packet+Flow', 'alpha': 1.0, 'beta': 1.0, 'gamma': 0.0},
        {'name': 'Flow+Campaign', 'alpha': 0.0, 'beta': 1.0, 'gamma': 1.0},
    ]

    ablation_results = {}

    for config in configurations:
        print(f"\nTesting: {config['name']}")
        print("-" * 60)

        # Get dimensions
        sample = dataset_train[0]
        packet_dim = len(sample['packet'])
        flow_seq_len = len(sample['flow'])
        campaign_dim = len(sample['campaign'])

        # Initialize model
        model = MALZDA(
            packet_dim=packet_dim,
            flow_seq_len=flow_seq_len,
            campaign_dim=campaign_dim,
            embedding_dim=DEFAULT_EMBEDDING_DIM
        )

        # Set distance weights
        model.alpha.data = torch.tensor(config['alpha'])
        model.beta.data = torch.tensor(config['beta'])
        model.gamma.data = torch.tensor(config['gamma'])

        # Run experiment
        try:
            _, eval_results, _, _ = run_experiment(
                dataset_train, dataset_test,
                n_way=n_way, k_shot=k_shot, n_query=n_query,
                num_episodes=num_episodes, eval_episodes=eval_episodes,
                use_compositional=True,
                experiment_name=f"ablation_{config['name'].replace(' ', '_').lower()}"
            )

            ablation_results[config['name']] = {
                'accuracy': np.mean(eval_results['accuracies']),
                'accuracy_std': np.std(eval_results['accuracies']),
                'f1': np.mean(eval_results['f1_scores']),
                'f1_std': np.std(eval_results['f1_scores'])
            }

            print(
                f"Accuracy: {ablation_results[config['name']]['accuracy']:.4f} ± {ablation_results[config['name']]['accuracy_std']:.4f}")
            print(
                f"F1 Score: {ablation_results[config['name']]['f1']:.4f} ± {ablation_results[config['name']]['f1_std']:.4f}")

        except Exception as e:
            print(f"Error in configuration {config['name']}: {str(e)}")
            continue

    # Visualize results
    if ablation_results:
        visualize_ablation_results(ablation_results)

    # Save results
    with open(RESULTS_DIR / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


################################################
# SECTION 10: MAIN EXECUTION                   #
################################################

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("MAL-ZDA: Multi-level Adaptive Learning for Zero-Day Attack Detection")
    print("Hierarchical Few-Shot Learning Framework")
    print("="*80)

    # Configuration
    config = {
        'n_way': DEFAULT_N_WAY,
        'k_shot': DEFAULT_K_SHOT,
        'n_query': DEFAULT_N_QUERY,
        'num_episodes': 1000,
        'eval_episodes': 200,
        'use_compositional': True
    }

    # Check for real dataset
    data_files = list(DATA_DIR.glob("*.csv"))

    if len(data_files) > 0:
        print(f"\nFound {len(data_files)} CSV file(s) in dataset directory")
        print("Using real dataset...")

        # Load real data
        data_path = data_files[0]
        print(f"Loading: {data_path.name}")

        try:
            X_scaled, X_unscaled, y, feature_names, kill_chain_labels = load_and_preprocess_real_data(
                data_path)

            # Split into train and test
            X_train, X_test, y_train, y_test, kc_train, kc_test = train_test_split(
                X_scaled, y, kill_chain_labels,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )

            print(
                f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

            # Create datasets
            dataset_train = CyberSecurityDataset(
                X=X_train,
                y=y_train,
                kill_chain_labels=kc_train,
                feature_names=feature_names
            )

            dataset_test = CyberSecurityDataset(
                X=X_test,
                y=y_test,
                kill_chain_labels=kc_test,
                feature_names=feature_names
            )

        except Exception as e:
            print(f"\nError loading real data: {str(e)}")
            print("Falling back to synthetic data generation...")

            dataset_train = CyberSecurityDataset(
                num_classes=15,
                samples_per_class=1000,
                feature_dim=256,
                temporal_length=100,
                mode='train'
            )

            dataset_test = CyberSecurityDataset(
                num_classes=10,
                samples_per_class=500,
                feature_dim=256,
                temporal_length=100,
                mode='test'
            )

    else:
        print("\nNo CSV files found in dataset directory")
        print("Generating synthetic data...")

        dataset_train = CyberSecurityDataset(
            num_classes=15,
            samples_per_class=1000,
            feature_dim=256,
            temporal_length=100,
            mode='train'
        )

        dataset_test = CyberSecurityDataset(
            num_classes=10,
            samples_per_class=500,
            feature_dim=256,
            temporal_length=100,
            mode='test'
        )

    # EXPERIMENT 1: Compositional vs Standard Sampling
    print("\n" + "="*80)
    print("EXPERIMENT 1: Compositional vs Standard Sampling Comparison")
    print("="*80)

    print("\n### Running with Compositional Kill-Chain Sampling ###")
    model_comp, results_comp, train_losses_comp, train_accs_comp = run_experiment(
        dataset_train, dataset_test,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        num_episodes=config['num_episodes'],
        eval_episodes=config['eval_episodes'],
        use_compositional=True,
        experiment_name="compositional"
    )

    visualize_training_results(
        train_losses_comp, train_accs_comp, results_comp,
        save_prefix="compositional"
    )

    print("\n### Running with Standard Sampling ###")
    model_std, results_std, train_losses_std, train_accs_std = run_experiment(
        dataset_train, dataset_test,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        num_episodes=config['num_episodes'],
        eval_episodes=config['eval_episodes'],
        use_compositional=False,
        experiment_name="standard"
    )

    visualize_training_results(
        train_losses_std, train_accs_std, results_std,
        save_prefix="standard"
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    comp_acc = np.mean(results_comp['accuracies'])
    std_acc = np.mean(results_std['accuracies'])
    improvement = (comp_acc - std_acc) * 100

    print(f"\nCompositional Sampling:")
    print(
        f"  Accuracy:  {comp_acc:.4f} ± {np.std(results_comp['accuracies']):.4f}")
    print(
        f"  F1 Score:  {np.mean(results_comp['f1_scores']):.4f} ± {np.std(results_comp['f1_scores']):.4f}")

    print(f"\nStandard Sampling:")
    print(
        f"  Accuracy:  {std_acc:.4f} ± {np.std(results_std['accuracies']):.4f}")
    print(
        f"  F1 Score:  {np.mean(results_std['f1_scores']):.4f} ± {np.std(results_std['f1_scores']):.4f}")

    print(f"\nImprovement: {improvement:+.2f}%")
    print("="*80)

    create_comparison_visualization(
        results_comp, results_std, "compositional_vs_standard"
    )

    # EXPERIMENT 2: Ablation Study
    print("\n" + "="*80)
    print("EXPERIMENT 2: Ablation Study")
    print("="*80)

    ablation_results = run_ablation_study(
        dataset_train, dataset_test,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        num_episodes=500,
        eval_episodes=100
    )

    # EXPERIMENT 3: Few-Shot Scaling
    print("\n" + "="*80)
    print("EXPERIMENT 3: FEW-SHOT SCALING EXPERIMENT")
    print("="*80)

    k_shot_values = [1, 3, 5, 10]
    scaling_results = {}

    for k_shot in k_shot_values:
        print(f"\n### Testing {k_shot}-shot learning ###")
        print("-" * 60)

        try:
            _, eval_results, _, _ = run_experiment(
                dataset_train, dataset_test,
                n_way=config['n_way'],
                k_shot=k_shot,
                n_query=config['n_query'],
                num_episodes=800,
                eval_episodes=150,
                use_compositional=True,
                experiment_name=f"scaling_{k_shot}shot"
            )

            scaling_results[k_shot] = {
                'accuracy': np.mean(eval_results['accuracies']),
                'accuracy_std': np.std(eval_results['accuracies']),
                'f1': np.mean(eval_results['f1_scores']),
                'f1_std': np.std(eval_results['f1_scores']),
                'precision': np.mean(eval_results['precisions']),
                'recall': np.mean(eval_results['recalls'])
            }

            print(f"Results for {k_shot}-shot:")
            print(
                f"  Accuracy: {scaling_results[k_shot]['accuracy']:.4f} ± {scaling_results[k_shot]['accuracy_std']:.4f}")
            print(
                f"  F1 Score: {scaling_results[k_shot]['f1']:.4f} ± {scaling_results[k_shot]['f1_std']:.4f}")

        except Exception as e:
            print(f"Error in {k_shot}-shot experiment: {str(e)}")
            continue

    # Visualize scaling results
    if len(scaling_results) > 0:
        visualize_scaling_results(scaling_results)

    # Generate final summary report
    generate_summary_report(
        results_comp, results_std, ablation_results, scaling_results
    )

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - Model checkpoints: results_malzda/*_model.pt")
    print("  - Experimental results: results_malzda/*_results.json")
    print("  - Visualizations:")
    print("    * compositional_results.png")
    print("    * standard_results.png")
    print("    * compositional_vs_standard.png")
    print("    * ablation.png")
    print("    * scaling.png")
    print("  - Summary report: results_malzda/summary_report.txt")
    print("="*80)


def generate_summary_report(
    results_comp: Dict,
    results_std: Dict,
    ablation_results: Dict,
    scaling_results: Dict
):
    """Generate comprehensive summary report"""

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MAL-ZDA EXPERIMENTAL RESULTS SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")

    # Compositional vs Standard
    report_lines.append("1. COMPOSITIONAL vs STANDARD SAMPLING")
    report_lines.append("-"*80)
    report_lines.append("")

    comp_acc = np.mean(results_comp['accuracies'])
    comp_f1 = np.mean(results_comp['f1_scores'])
    std_acc = np.mean(results_std['accuracies'])
    std_f1 = np.mean(results_std['f1_scores'])

    report_lines.append(f"Compositional Sampling:")
    report_lines.append(
        f"  Accuracy:  {comp_acc:.4f} ± {np.std(results_comp['accuracies']):.4f}")
    report_lines.append(
        f"  F1 Score:  {comp_f1:.4f} ± {np.std(results_comp['f1_scores']):.4f}")
    report_lines.append(
        f"  Precision: {np.mean(results_comp['precisions']):.4f} ± {np.std(results_comp['precisions']):.4f}")
    report_lines.append(
        f"  Recall:    {np.mean(results_comp['recalls']):.4f} ± {np.std(results_comp['recalls']):.4f}")
    report_lines.append("")

    report_lines.append(f"Standard Sampling:")
    report_lines.append(
        f"  Accuracy:  {std_acc:.4f} ± {np.std(results_std['accuracies']):.4f}")
    report_lines.append(
        f"  F1 Score:  {std_f1:.4f} ± {np.std(results_std['f1_scores']):.4f}")
    report_lines.append(
        f"  Precision: {np.mean(results_std['precisions']):.4f} ± {np.std(results_std['precisions']):.4f}")
    report_lines.append(
        f"  Recall:    {np.mean(results_std['recalls']):.4f} ± {np.std(results_std['recalls']):.4f}")
    report_lines.append("")

    improvement = (comp_acc - std_acc) * 100
    report_lines.append(f"Improvement: {improvement:+.2f}%")
    report_lines.append("")

    # Ablation Study
    if ablation_results:
        report_lines.append("2. ABLATION STUDY RESULTS")
        report_lines.append("-"*80)
        report_lines.append("")

        for config_name, results in ablation_results.items():
            report_lines.append(f"{config_name}:")
            report_lines.append(
                f"  Accuracy: {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
            report_lines.append(
                f"  F1 Score: {results['f1']:.4f} ± {results['f1_std']:.4f}")
            report_lines.append("")

    # Scaling Results
    if scaling_results:
        report_lines.append("3. FEW-SHOT SCALING RESULTS")
        report_lines.append("-"*80)
        report_lines.append("")

        for k_shot, results in scaling_results.items():
            report_lines.append(f"{k_shot}-shot Learning:")
            report_lines.append(
                f"  Accuracy:  {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
            report_lines.append(
                f"  F1 Score:  {results['f1']:.4f} ± {results['f1_std']:.4f}")
            report_lines.append(f"  Precision: {results['precision']:.4f}")
            report_lines.append(f"  Recall:    {results['recall']:.4f}")
            report_lines.append("")

    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Save report
    report_path = RESULTS_DIR / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nSummary report saved to {report_path}")

    # Print to console
    print("\n" + '\n'.join(report_lines))


################################################
# SECTION 11: ENTRY POINT                      #
################################################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExperiment terminated")


################################################
# END OF FILE                                  #
