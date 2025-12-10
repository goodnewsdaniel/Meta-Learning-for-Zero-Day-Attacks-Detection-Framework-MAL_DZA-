# MAL-ZDA: Multi-level Adaptive Learning for Zero-Day Attack Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## A Hierarchical Few-Shot Learning Framework for Cybersecurity Threat Detection

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Baseline Comparisons](#baseline-comparisons)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Results Interpretation](#results-interpretation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contact](#contact)

---

## 🎯 Overview

MAL-ZDA is a novel deep learning framework designed for zero-day attack detection in cybersecurity applications. It employs hierarchical few-shot learning with compositional task sampling to detect novel cyber threats with minimal labeled examples.

### Problem Statement

Traditional machine learning models struggle with:

- **Zero-day attacks**: Novel threats with no prior training examples
- **Limited labeled data**: Expensive and time-consuming annotation
- **Evolving attack patterns**: Rapidly changing threat landscape
- **Hierarchical attack structures**: Multi-stage kill-chain attacks

### Solution

MAL-ZDA addresses these challenges through:

- **Hierarchical encoding**: Multi-level feature extraction (packet → flow → campaign)
- **Few-shot learning**: Learn from 1-10 examples per class
- **Compositional sampling**: Kill-chain aware task generation
- **Prototypical networks**: Distance-based classification

---

## ✨ Key Features

### 🏗️ Hierarchical Architecture

- **Packet-level**: 1D-CNN for low-level feature extraction
- **Flow-level**: Bi-LSTM for temporal sequence modeling
- **Campaign-level**: Deep MLP for high-level pattern recognition

### 🎓 Few-Shot Learning

- Episodic meta-learning paradigm
- Support for 1-shot to 10-shot learning
- Prototypical network classification
- Learnable distance weighting

### 🔗 Compositional Task Sampling

- Kill-chain phase awareness (Recon → Exploit → C2 → Exfiltration)
- Phase-diverse support sets
- Realistic attack scenario simulation

### 📊 Comprehensive Evaluation

- Multiple experimental protocols (3 standard experiments)
- Ablation studies (hierarchical component analysis)
- Statistical significance testing
- Rich visualizations

### 🔍 Baseline Comparisons

- 5 baseline implementations for empirical validation
- Supervised CNN-LSTM (upper bound)
- One-Class SVM (anomaly detection)
- Transfer Learning (pre-train + fine-tune)
- MAML (optimization-based meta-learning)
- Prototypical Networks (generic metric-learning)
- Publication-ready comparative analysis

### 🔄 Dual Data Support

- **Real datasets**: CSV file processing with robust preprocessing
- **Synthetic data**: Automatic generation if no dataset provided
- Seamless fallback mechanism

---

## 🏛️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     MAL-ZDA Architecture                     │
└─────────────────────────────────────────────────────────────┘

Input Data (Network Traffic)
        │
        ├─────────────────┬──────────────────┬─────────────────┐
        │                 │                  │                 │
        ▼                 ▼                  ▼                 │
┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   Packet     │  │     Flow     │  │   Campaign   │         │
│   Features   │  │   Features   │  │   Features   │         │
│              │  │              │  │              │         │
│   [F₁...Fₙ]  │  │  [T₁...Tₘ]   │  │  Aggregated  │         │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
       │                 │                  │                 │
       ▼                 ▼                  ▼                 │
┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   1D-CNN     │  │  Bi-LSTM     │  │  Deep MLP    │         │
│              │  │              │  │              │         │
│  • Conv1D    │  │  • Forward   │  │  • FC Layers │         │
│  • BatchNorm │  │  • Backward  │  │  • BatchNorm │         │
│  • ReLU      │  │  • Dropout   │  │  • Dropout   │         │
│  • MaxPool   │  │              │  │              │         │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
       │                 │                  │                 │
       │                 │                  │                 │
       ▼                 ▼                  ▼                 │
┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   Embedding  │  │   Embedding  │  │   Embedding  │         │
│   E_packet   │  │    E_flow    │  │  E_campaign  │         │
│   [128-dim]  │  │   [128-dim]  │  │   [128-dim]  │         │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
       │                 │                  │                 │
       └─────────────────┴──────────────────┘                 │
                         │                                    │
                         ▼                                    │
              ┌──────────────────────┐                        │
              │  Hierarchical Fusion │                        │
              │                      │                        │
              │  d = α·d_p + β·d_f  │                        │
              │      + γ·d_c         │                        │
              └──────────┬───────────┘                        │
                         │                                    │
                         ▼                                    │
              ┌──────────────────────┐                        │
              │   Prototype-based    │                        │
              │   Classification     │                        │
              │                      │                        │
              │  Class = argmin(d)   │                        │
              └──────────────────────┘                        │
                                                              │
                    Kill-Chain Phase Labels ─────────────────┘
                    (Recon, Exploit, C2, Exfil)
```

### Component Details

#### 1. Hierarchical Encoder

- **Input**: Raw network traffic features
- **Output**: Multi-level embeddings (128-dim each)
- **Trainable Parameters**: ~2M

#### 2. Distance Computation

- **Metric**: Weighted Euclidean distance
- **Learnable Weights**: α (packet), β (flow), γ (campaign)
- **Temperature Scaling**: τ for calibration

#### 3. Prototypical Classification

- **Support Set**: K examples per class (K=1,3,5,10)
- **Prototype**: Mean embedding per class
- **Query Classification**: Nearest prototype

---

## 🔧 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Step 1: Clone Repository

```bash
git clone https://github.com/goodnewsdaniel/Meta-Learning-for-Zero-Day-Attacks-Detection-Framework-MAL_DZA.git
cd malzda
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv malzda_env
source malzda_env/bin/activate  # On Windows: malzda_env\Scripts\activate

# Or using conda
conda create -n malzda python=3.8
conda activate malzda
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### requirements.txt

```txt
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
scipy>=1.7.0
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 Dataset Preparation

### Option 1: Using Your Own Dataset (Recommended)

#### Dataset Format

Your CSV file should follow this structure:

```csv
feature_1,feature_2,...,feature_n,target
0.123,0.456,...,0.789,0
0.234,0.567,...,0.890,1
...
```

**Requirements:**

- Last column must be the target/label
- All other columns are features
- Features can be numeric or categorical
- Missing values will be handled automatically

#### Placement

```bash
# Create dataset directory
mkdir dataset

# Copy your CSV file
cp /path/to/your/data.csv dataset/

# Example structure
dataset/
└── 5G_data_merged.csv
```

#### Supported Dataset Types

1. **Network Traffic Data**
   - Packet captures (PCAP-derived features)
   - NetFlow/IPFIX records
   - 5G network logs

2. **Intrusion Detection Datasets**
   - NSL-KDD
   - CICIDS2017/2018
   - UNSW-NB15
   - Custom datasets

3. **IoT Security Data**
   - Device telemetry
   - Protocol anomalies
   - Botnet traffic

### Option 2: Using Synthetic Data (Automatic)

If no CSV file is found, the system automatically generates synthetic data:

```python
# Automatic generation with realistic characteristics
- 15 attack classes
- 1000 samples per class
- 256-dimensional features
- Temporal sequences (100 timesteps)
- Kill-chain phase labels
```

### Data Preprocessing Pipeline

The framework automatically performs:

1. **Column Name Standardization**

   ```python
   "Source IP" → "Source_Ip"
   "dest port" → "Dest_Port"
   ```

2. **Outlier Handling**

   - Clip to 1st-99th percentiles
   - Remove infinite values

3. **Missing Value Imputation**

   - Numeric: Median imputation
   - Categorical: Mode imputation

4. **Feature Encoding**

   - Categorical: Label encoding
   - Numeric: Standard scaling (z-score)

5. **Class Balancing**

   - Remove classes with < 2 samples
   - Stratified train-test split

6. **Kill-Chain Labeling**

   - Automatic phase assignment
   - Based on class distribution

---

## 🚀 Usage

### Basic Usage

```bash
# Run with default configuration
python mal_zda.py
```

### Output

```text
================================================================================
MAL-ZDA: Multi-level Adaptive Learning for Zero-Day Attack Detection
Hierarchical Few-Shot Learning Framework
================================================================================

Using device: cuda

Loading data from: dataset/5G_data_merged.csv
Initial data shape: (50000, 85)

Data Processing Summary:
------------------------------------------------------------
Final shape: X=(50000, 84), y=(50000,)
Features: 84
Classes: 15
Class distribution: {0: 5234, 1: 4891, ..., 14: 2156}
Kill-chain distribution: {0: 15234, 1: 12456, 2: 10234, 3: 7845, 4: 4231}

Train set: (40000, 84), Test set: (10000, 84)
Loaded 40000 real samples with 15 classes
Loaded 10000 real samples with 15 classes

================================================================================
EXPERIMENT 1: Compositional vs Standard Sampling Comparison
================================================================================

### Running with Compositional Kill-Chain Sampling ###
Training for 1000 episodes...
Episode 100: Loss=0.5234, Accuracy=0.8456
  Weights: α=0.856, β=1.123, γ=0.987
Episode 200: Loss=0.3456, Accuracy=0.9012
  Weights: α=0.923, β=1.056, γ=1.034
...

================================================================================
FINAL RESULTS
================================================================================
Test Accuracy:  0.9234 ± 0.0156
Test F1 Score:  0.9123 ± 0.0178
Test Precision: 0.9267 ± 0.0145
Test Recall:    0.9189 ± 0.0162
================================================================================
```

### Advanced Usage

#### Custom Configuration

```python
# Edit main() function configuration
config = {
    'n_way': 5,           # 5-way classification
    'k_shot': 1,          # 1-shot learning
    'n_query': 15,        # 15 query samples per class
    'num_episodes': 1000, # Training episodes
    'eval_episodes': 200, # Evaluation episodes
    'use_compositional': True  # Use kill-chain sampling
}
```

#### Run Specific Experiments

```python
# Run only compositional experiment
python -c "from mal_zda import *; 
dataset = CyberSecurityDataset(); 
run_experiment(dataset, dataset, experiment_name='my_experiment')"
```

#### Load and Evaluate Saved Model

```python
import torch
from mal_zda import MALZDA, MALZDATrainer

# Load model
model = MALZDA(packet_dim=256, embedding_dim=128)
trainer = MALZDATrainer(model)
trainer.load_model('results_malzda/compositional_model.pt')

# Evaluate on new data
results = trainer.evaluate_episode(support_set, query_set, 
                                   support_labels, query_labels)
print(f"Accuracy: {results['accuracy']:.4f}")
```

---

## 🧪 Experiments

### Experiment 1: Compositional vs Standard Sampling

**Purpose**: Evaluate the impact of kill-chain aware task sampling

**Configuration:**

- N-way: 5
- K-shot: 1
- Episodes: 1000 (train), 200 (test)

**Outputs:**

- `compositional_results.png`
- `standard_results.png`
- `compositional_vs_standard.png`

**Expected Results:**

- Compositional sampling: +5-10% accuracy improvement
- Better generalization to novel attack patterns

### Experiment 2: Ablation Study

**Purpose**: Assess contribution of each hierarchical level

**Configurations:**

1. Full Model (α=1, β=1, γ=1)
2. Packet Only (α=1, β=0, γ=0)
3. Flow Only (α=0, β=1, γ=0)
4. Campaign Only (α=0, β=0, γ=1)
5. Packet+Flow (α=1, β=1, γ=0)
6. Flow+Campaign (α=0, β=1, γ=1)

**Outputs:**

- `ablation.png`
- `ablation_results.json`

**Expected Results:**

- Full model achieves highest performance
- Flow level contributes most (temporal patterns)
- Campaign level provides context

### Experiment 3: Few-Shot Scaling

**Purpose**: Evaluate performance across different shot values

**Configurations:**

- 1-shot, 3-shot, 5-shot, 10-shot
- N-way: 5
- Episodes: 800 (train), 150 (test)

**Outputs:**

- `scaling.png`
- `scaling_*shot_results.json`

**Expected Results:**

- Performance increases with more shots
- Diminishing returns after 5-shot
- Strong 1-shot performance (~85% accuracy)

---

## 📊 Baseline Comparisons

### Overview

MAL-ZDA is evaluated against 5 diverse baseline approaches to demonstrate empirical advantages:

### Baseline 1: Supervised CNN-LSTM (Upper Bound)

**Description**: Full supervision on all classes (not applicable to few-shot setting)

**Architecture:**

- CNN for packet-level features
- LSTM for flow-level sequences
- Full classification head

**Characteristics:**

- Maximum possible performance
- Requires labeled data for all classes
- Impractical for zero-day detection

**Expected Accuracy:** ~95% (upper bound)

### Baseline 2: One-Class SVM (Anomaly Detection)

**Description**: Unsupervised anomaly detection approach

**Architecture:**

- scikit-learn OneClassSVM
- RBF kernel
- Binary classification (normal vs anomaly)

**Characteristics:**

- Unsupervised learning
- Good for anomaly detection
- Limited to binary classification

**Expected Accuracy:** ~70% (anomaly detection task)

### Baseline 3: Transfer Learning (Pre-train + Fine-tune)

**Description**: Leverage pre-training on base classes, fine-tune on few-shot

**Architecture:**

- Pre-trained feature extractor
- Learnable classification head
- Gradual unfreezing strategy

**Characteristics:**

- Leverages available data efficiently
- Practical for real-world scenarios
- Requires diverse base classes

**Expected Accuracy:** ~80%

### Baseline 4: MAML (Model-Agnostic Meta-Learning)

**Description**: Optimization-based meta-learning competitor

**Architecture:**

- Learnable feature extractor
- Inner/outer loop optimization
- Adaptive gradient-based learning

**Characteristics:**

- State-of-the-art meta-learning approach
- Competitive with MAL-ZDA
- Demonstrates value of architectural design

**Expected Accuracy:** ~82%

### Baseline 5: Prototypical Networks (Generic Metric-Learning)

**Description**: Simple metric-learning baseline without hierarchy

**Architecture:**

- Feature extractor (no hierarchy)
- Distance-based classification
- Learnable embeddings

**Characteristics:**

- Non-hierarchical approach
- Shows value of hierarchical design
- Simpler than MAL-ZDA

**Expected Accuracy:** ~75%

### Comparative Results

| Baseline | Accuracy | F1 Score | Key Advantage |
|----------|----------|----------|---------------|
| Supervised CNN-LSTM | ~95% | ~93% | Upper bound (full supervision) |
| MAML | ~82% | ~80% | Competitive meta-learning |
| Transfer Learning | ~80% | ~78% | Pre-training leverage |
| Prototypical Networks | ~75% | ~73% | Simple metric-learning |
| One-Class SVM | ~70% | ~68% | Anomaly detection |
| **MAL-ZDA (Proposed)** | **~88%** | **~85%** | **Hierarchical + Meta-learning** |

### Baseline Implementation Details

All baselines are implemented in the same framework with:

- Consistent data preprocessing
- Identical evaluation protocols
- Same train/test splits
- Unified result reporting

See the [Baseline Comparisons Documentation](BASELINES_QUICK_START.md) for detailed implementation information.

---

## 📊 Output Structure

```text
project_root/
├── mal_zda.py                          # Main implementation
├── dataset/                            # Data directory
│   └── your_data.csv                   # Your dataset (optional)
├── results_malzda/                     # Output directory
│   ├── compositional_model.pt          # Trained model
│   ├── compositional_results.json      # Detailed results
│   ├── compositional_results.png       # Visualization
│   ├── standard_model.pt
│   ├── standard_results.json
│   ├── standard_results.png
│   ├── compositional_vs_standard.png   # Comparison
│   ├── ablation_*.pt                   # Ablation models
│   ├── ablation_results.json
│   ├── ablation.png
│   ├── scaling_*.pt                    # Scaling models
│   ├── scaling.png
│   ├── baselines_comparison.png        # Baseline comparison (NEW)
│   ├── baselines_results.json          # Baseline results (NEW)
│   └── summary_report.txt              # Comprehensive report
└── README.md                           # This file
```

### File Descriptions

#### Model Checkpoints (.pt)

Contains:

- Model state dictionary
- Optimizer state
- Training history

#### Results Files (.json)

Contains:

- Configuration parameters
- Training metrics (loss, accuracy)
- Evaluation metrics (accuracy, F1, precision, recall)
- Per-episode results

#### Visualization Files (.png)

1. **Training Results**
   - Loss curve with moving average
   - Accuracy curve with moving average
   - Metrics distribution (boxplot)
   - Accuracy histogram
   - Confusion matrix
   - Performance summary table

2. **Comparison Plots**
   - Side-by-side boxplots
   - Precision-recall scatter
   - Mean metrics bar chart

3. **Ablation Plots**
   - Configuration comparison
   - Accuracy and F1 scores

4. **Scaling Plots**
   - Performance vs support size
   - All metrics trends

#### Summary Report (.txt)

Comprehensive text report with:

- All experimental results
- Statistical comparisons
- Key findings
- Performance tables

---

## ⚙️ Configuration

### Model Hyperparameters

```python
# In main() function
DEFAULT_N_WAY = 5              # Number of classes per episode
DEFAULT_K_SHOT = 1             # Support examples per class
DEFAULT_N_QUERY = 15           # Query examples per class
DEFAULT_EMBEDDING_DIM = 128    # Embedding dimension

# Training
learning_rate = 0.001          # Adam optimizer learning rate
num_episodes = 1000            # Training episodes
eval_episodes = 200            # Evaluation episodes
```

### Architecture Parameters

```python
# Hierarchical encoder
packet_dim = 256               # Packet feature dimension
flow_seq_len = 100             # Temporal sequence length
campaign_dim = 1024            # Campaign feature dimension (4x packet_dim)
embedding_dim = 128            # Output embedding dimension

# CNN (Packet encoder)
channels = [1, 32, 64, 128]    # Convolutional channels
kernel_size = 5                # Convolution kernel
dropout = 0.2                  # Dropout rate

# LSTM (Flow encoder)
hidden_size = 64               # LSTM hidden units
num_layers = 2                 # LSTM layers
bidirectional = True           # Use Bi-LSTM

# MLP (Campaign encoder)
hidden_layers = [512, 256, 128] # MLP hidden dimensions
dropout = [0.3, 0.2, 0]        # Per-layer dropout
```

### Data Preprocessing

```python
TEST_SIZE = 0.2                # Train-test split ratio
RANDOM_STATE = 42              # Random seed
N_BINS = 10                    # Discretization bins

# Outlier handling
lower_percentile = 0.01        # Lower clip threshold
upper_percentile = 0.99        # Upper clip threshold
```

### Device Configuration

```python
# Automatic GPU detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Force CPU usage
DEVICE = torch.device('cpu')

# Specific GPU
DEVICE = torch.device('cuda:0')
```

---

## 📈 Results Interpretation

### Performance Metrics

#### Accuracy

- **Definition**: Proportion of correct predictions
- **Range**: [0, 1], higher is better
- **Interpretation**:
  - > 0.90: Excellent
  - 0.80-0.90: Good
  - 0.70-0.80: Fair
  - < 0.70: Needs improvement

#### F1 Score

- **Definition**: Harmonic mean of precision and recall
- **Range**: [0, 1], higher is better
- **Use**: Balanced metric for imbalanced classes

#### Precision

- **Definition**: True Positives / (True Positives + False Positives)
- **Interpretation**: How many predicted attacks are actual attacks
- **Important for**: Minimizing false alarms

#### Recall

- **Definition**: True Positives / (True Positives + False Negatives)
- **Interpretation**: How many actual attacks are detected
- **Important for**: Minimizing missed attacks

### Distance Weights

The learned hierarchical weights indicate importance:

```python
α (alpha)   - Packet-level importance
β (beta)    - Flow-level importance
γ (gamma)   - Campaign-level importance
```

**Interpretation:**

- **β > α, γ**: Temporal patterns most important
- **γ > α, β**: High-level context most important
- **Balanced (α ≈ β ≈ γ)**: All levels contribute equally

### Typical Results

#### Good Performance Indicators

- ✅ Accuracy > 0.85
- ✅ F1 Score > 0.82
- ✅ Low standard deviation (< 0.02)
- ✅ Compositional > Standard by 5%+
- ✅ Smooth training curves

#### Potential Issues

- ⚠️ Accuracy < 0.70
- ⚠️ High variance (std > 0.05)
- ⚠️ Overfitting (train >> test)
- ⚠️ Unstable training (oscillating loss)

### Statistical Significance

Results include confidence intervals (mean ± std):

- **Non-overlapping CIs**: Statistically significant difference
- **Overlapping CIs**: No significant difference

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error

**Error:**

```text
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Reduce batch size
BATCH_SIZE = 16  # Default: 32

# Reduce embedding dimension
DEFAULT_EMBEDDING_DIM = 64  # Default: 128

# Use CPU instead
DEVICE = torch.device('cpu')
```

#### 2. Dataset Loading Error

**Error:**

```text
Error loading data: FileNotFoundError
```

**Solutions:**

```bash
# Check dataset directory
ls dataset/

# Ensure CSV file exists
# Or let it generate synthetic data automatically
```

#### 3. Dimension Mismatch

**Error:**

```text
RuntimeError: size mismatch
```

**Solutions:**

- Ensure all features are numeric
- Check for NaN values in data
- Verify consistent feature dimensions

#### 4. Poor Performance

**Symptoms:**

- Accuracy < 0.60
- High variance

**Solutions:**

```python
# Increase training episodes
num_episodes = 2000  # Default: 1000

# Adjust learning rate
learning_rate = 0.0001  # Default: 0.001

# Try different k-shot values
k_shot = 5  # Default: 1

# Check data quality
# - Sufficient samples per class (>100)
# - Balanced class distribution
# - Relevant features
```

#### 5. Slow Training

**Solutions:**

```python
# Enable GPU
# Install CUDA-enabled PyTorch

# Reduce sequence length
temporal_length = 50  # Default: 100

# Reduce evaluation frequency
# Comment out intermediate evaluations
```

#### 6. Import Errors

**Error:**

```text
ModuleNotFoundError: No module named 'torch'
```

**Solutions:**

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+

# Verify virtual environment
which python
```

---

## 📚 Citation

If you use MAL-ZDA in your research, please cite:

```bibtex
@article{daniel2025malzda,
  title={MAL-ZDA: Multi-level Adaptive Learning for Zero-Day Attack Detection},
  author={Daniel, Goodnews},
  journal={University of Johannesburg},
  year={2025},
  department={Electrical \& Electronics Engineering},
  note={PhD Research}
}
```

---

## 📧 Contact

**Author**: Goodnews Daniel (PhD Candidate)  
**Email**: 222166453@student.uj.ac.za  
**Institution**: University of Johannesburg  
**Department**: Electrical & Electronics Engineering  
**Faculty**: Engineering & the Built Environment

### Support

For questions, issues, or contributions:

1. **Documentation Files**
   - [BASELINES_QUICK_START.md](BASELINES_QUICK_START.md) - Quick reference for baseline comparisons
   - [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - Detailed baseline implementation documentation
   - [CODE_NAVIGATION_GUIDE.md](CODE_NAVIGATION_GUIDE.md) - Code structure and navigation guide
   - [README_INTEGRATION.md](README_INTEGRATION.md) - Integration overview

2. **Support Channels**
   - Open an issue on GitHub
   - Email the author
   - Check documentation files

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- University of Johannesburg
- Department of Electrical & Electronics Engineering
- Cybersecurity Research Group
- Open-source community (PyTorch, scikit-learn, etc.)

---

## 🔄 Version History

### v1.1.0 (Current - December 2025)

- **NEW**: 5 baseline implementations for comprehensive comparison
  - Supervised CNN-LSTM (upper bound)
  - One-Class SVM (anomaly detection)
  - Transfer Learning (pre-train + fine-tune)
  - MAML (optimization-based meta-learning)
  - Prototypical Networks (generic metric-learning)
- **NEW**: Experiment 4 - Baseline Comparison module
- **NEW**: Comprehensive baseline documentation
- **Enhanced**: Summary report includes baseline comparison
- **Enhanced**: Visualization pipeline expanded for baselines
- Backward compatible with v1.0.0

### v1.0.0 (November 2025)

- Initial release
- Hierarchical encoder implementation
- Compositional task sampling
- Comprehensive experimental framework (3 experiments)
- Dual data support (real/synthetic)
- Complete visualization suite
- Ablation study framework
- Few-shot scaling experiments

### Planned Features

- [ ] Multi-GPU support
- [ ] Real-time inference module
- [ ] Web-based dashboard
- [ ] Additional datasets support
- [ ] Automated hyperparameter tuning
- [ ] Distributed training support

---

**Last Updated**: December 2025  
**Status**: Production Ready ✅
