# MAL-ZDA Code Navigation Guide

## Quick Navigation Map

Use this guide to find specific components in `mal_dza_algorithm.py`

---

## 📍 SECTION LOCATIONS

### SECTION 1: Imports & Constants (Lines 1-124)

```text
Line 15:  # SECTION 1: IMPORTS AND CONSTANTS
Line 73:  # SECTION 1.1: GLOBAL CONSTANTS
```

**Key Elements:**

- Lines 18-50: Standard library imports
- Lines 53-61: ML library imports
- Lines 75-95: Global constants definition
- Line 95-105: Device configuration

### SECTION 2: Data Loading (Lines 125-518)

```text
Line 125: # SECTION 2: DATA LOADING AND PREPROCESSING
```

**Key Elements:**

- Line 128: `def load_and_preprocess_real_data()`
- Line 303: `class CyberSecurityDataset(Dataset):`
- Line 351: `def _create_temporal_from_static()` ← FIX #1

### SECTION 3: Hierarchical Encoder (Lines 519-629)

```text
Line 519: # SECTION 3: HIERARCHICAL ENCODER ARCHITECTURE
```

**Key Elements:**

- Line 522: `class HierarchicalEncoder(nn.Module):`
- Line 580: Packet encoder (CNN)
- Line 596: Flow encoder (LSTM)
- Line 607: Campaign encoder (MLP)

### SECTION 4: MAL-ZDA Model (Lines 630-765)

```text
Line 630: # SECTION 4: MAL-ZDA MODEL IMPLEMENTATION
```

**Key Elements:**

- Line 633: `class MALZDA(nn.Module):`
- Line 659: `def _encode_batch()` ← FIX #4 (device placement)
- Line 686: `def compute_prototypes()`
- Line 706: `def compute_distances()`
- Line 740: `def get_positive_weights()` ← FIX #7

### SECTION 5: Task Sampling (Lines 766-980)

```text
Line 766: # SECTION 5: COMPOSITIONAL TASK SAMPLING (FIXED)
```

**Key Elements:**

- Line 769: `class CompositionalTaskSampler:`
- Line 805: `def _build_indices()`
- Line 820: `def sample_task()`
- Line 826: `def _sample_compositional_task()`
- Line 896: `def _sample_standard_task()` ← FIX #1

### SECTION 6: Training & Evaluation (Lines 981-1188)

```text
Line 981: # SECTION 6: TRAINING AND EVALUATION (FIXED)
```

**Key Elements:**

- Line 984: `class MALZDATrainer:`
- Line 1006: `def _convert_batch_to_tensors()` ← FIX #2
- Line 1032: `def train_episode()` ← FIX #3 (NaN validation)
- Line 1093: `def evaluate_episode()`
- Line 1144: `def save_model()`

### SECTION 7: Experimental Framework (Lines 1189-1409)

```text
Line 1189: # SECTION 7: EXPERIMENTAL FRAMEWORK
```

**Key Elements:**

- Line 1192: `def run_experiment()` ← Main experiment runner
- Line 1395: `def run_ablation_study()`

### SECTION 8: Visualization (Lines 1410-2349)

```text
Line 1410: # SECTION 8: VISUALIZATION FUNCTIONS
```

**Key Elements:**

- Line 1413: `def _save_individual_scaling_all_metrics()`
- Line 1437: `def visualize_training_results()`
- Line 1815: `def create_comparison_visualization()`
- Line 2004: `def visualize_ablation_results()`
- Line 2072: `def visualize_scaling_results()`

### **SECTION 9: BASELINE IMPLEMENTATIONS** (Lines 2350-3024) ✅ **NEW**

```text
Line 2350: # SECTION 9: BASELINE IMPLEMENTATIONS
```

**Key Elements:**

- Line 2354: `class SupervisedCNNLSTM(nn.Module):`
- Line 2394: `class OneClassSVMBaseline:`
- Line 2418: `class TransferLearningBaseline(nn.Module):`
- Line 2471: `class MAMLBaseline(nn.Module):`
- Line 2528: `class PrototypicalNetworkBaseline(nn.Module):`
- Line 2572: `def train_supervised_baseline():`
- Line 2619: `def evaluate_baseline():`
- Line 2652: `def run_baseline_comparison():` ⭐ **MAIN FUNCTION**
- Line 2898: `def visualize_baseline_comparison():`

### SECTION 10: Main Execution (Lines 3025-3406)

```text
Line 3025: # SECTION 10: MAIN EXECUTION
Line 3028: def main():
```

**Key Experiments:**

- Line 3090: Experiment 1 (Compositional vs Standard)
- Line 3135: Experiment 2 (Ablation Study)
- Line 3163: Experiment 3 (Few-Shot Scaling)
- Line 3210: **Experiment 4 (Baseline Comparison)** ← **NEW**
- Line 3222: `def generate_summary_report():`

---

## 🔍 QUICK REFERENCE BY COMPONENT

### Finding Baseline Classes

```text
SupervisedCNNLSTM ............... Line 2354
OneClassSVMBaseline ............ Line 2394
TransferLearningBaseline ....... Line 2418
MAMLBaseline ................... Line 2471
PrototypicalNetworkBaseline .... Line 2528
```

### Finding Baseline Functions

```text
train_supervised_baseline() .... Line 2572
evaluate_baseline() ............ Line 2619
run_baseline_comparison() ...... Line 2652 ⭐
visualize_baseline_comparison() Line 2898
```

### Finding Data Classes

```text
CyberSecurityDataset .......... Line 303
HierarchicalEncoder ........... Line 522
MALZDA ........................ Line 633
CompositionalTaskSampler ...... Line 769
MALZDATrainer ................. Line 984
```

### Finding Critical Fixes

```text
FIX #1 (temporal_length) ....... Line 358
FIX #2 (_convert_batch_to_tensors) ... Line 1006
FIX #3 (NaN/Inf validation) .... Line 1050
FIX #4 (device placement) ...... Line 668
FIX #7 (positive weights) ...... Line 744
```

### Finding Experiments

```text
run_experiment() .............. Line 1192
run_ablation_study() .......... Line 1395
Experiment 1 (in main) ........ Line 3090
Experiment 2 (in main) ........ Line 3135
Experiment 3 (in main) ........ Line 3163
Experiment 4 (in main) ........ Line 3210 ✅ NEW
```

---

## 📊 FILE STATISTICS

### By Section

```text
SECTION 1:  Imports & Constants      Lines 1-124      (124 lines)
SECTION 2:  Data Loading             Lines 125-518    (394 lines)
SECTION 3:  Hierarchical Encoder     Lines 519-629    (111 lines)
SECTION 4:  MAL-ZDA Model            Lines 630-765    (136 lines)
SECTION 5:  Task Sampling            Lines 766-980    (215 lines)
SECTION 6:  Training & Evaluation    Lines 981-1188   (208 lines)
SECTION 7:  Experimental Framework   Lines 1189-1409  (221 lines)
SECTION 8:  Visualization            Lines 1410-2349  (940 lines)
SECTION 9:  BASELINES ✅ NEW          Lines 2350-3024  (675 lines)
SECTION 10: Main Execution           Lines 3025-3406  (382 lines)
────────────────────────────────────────────────────────
TOTAL:                                              3,406 lines
```

### Original vs Updated

```text
Original (SECTIONS 1-8): 2,716 lines
New (SECTION 9):         + 690 lines
Updated Total:           3,406 lines
Increase:                +25.4%
```

---

## 🎯 HOW TO USE THIS GUIDE

### To Find a Specific Class

```text
1. Look in "QUICK REFERENCE BY COMPONENT" section
2. Find the exact line number
3. Go to mal_dza_algorithm.py and Ctrl+G to that line
4. Read the class/function documentation
```

### To Understand the Flow

```text
1. Start with main() at line 3028
2. Follow the experiment sequence
3. For each experiment, look up the referenced function
4. Use line numbers to navigate
```

### To Find a Specific Fix

```text
1. Look in "Finding Critical Fixes" subsection
2. Go to the line number
3. Read the fix comment and code
4. Look at surrounding context
```

### To Run Individual Components

```python
from mal_dza_algorithm import CyberSecurityDataset, run_experiment

# Line 303: CyberSecurityDataset class
dataset = CyberSecurityDataset(num_classes=15, ...)

# Line 1192: run_experiment function
model, results, losses, accs = run_experiment(dataset_train, dataset_test)

# Line 2652: run_baseline_comparison function
baselines = run_baseline_comparison(dataset_train, dataset_test)
```

---

## 📍 NAVIGATION EXAMPLE

### Example: How to understand baseline comparison?

1. **Find the function** → Line 2652 (`run_baseline_comparison`)
2. **Read documentation** → Lines 2652-2670 (docstring)
3. **Understand flow** → Lines 2697-2895 (5 baseline implementations)
4. **See results** → Line 2898 (visualization function)
5. **Check main() integration** → Line 3210 (Experiment 4 call)
6. **Verify output** → Line 3221 (results saved)

### Example: How to find and understand a fix?

1. **Need to understand FIX #2?** → Line 1006
2. **Read the docstring** → Lines 1007-1012
3. **See the fix** → Lines 1013-1027
4. **Find where it's used** → Line 1044 (train_episode calls it)
5. **Understand the impact** → FIX #2 prevents data corruption

---

## 🚀 COMMON NAVIGATION TASKS

### "I want to modify the Supervised baseline"

→ Go to line 2354 (SupervisedCNNLSTM class)

### "I want to add another baseline"

→ Add new class after line 2569, update run_baseline_comparison() at line 2652

### "I want to change experiment parameters"

→ Go to line 3028 (main function), modify config dict at line 3046

### "I want to understand the hierarchical encoder"

→ Go to line 522 (HierarchicalEncoder class)

### "I want to see all the fixes applied"

→ Find FIX references: Lines 358, 1006, 1050, 668, 744

### "I want to run only baselines"

→ Import from line 2652 (`run_baseline_comparison`) and call directly

---

## 📞 DEBUGGING WITH THIS GUIDE

### Error in baseline training?

1. Check line 2572 (`train_supervised_baseline` function)
2. See line 1006 (tensor conversion - FIX #2)
3. Check line 1050 (NaN validation - FIX #3)

### Device compatibility issue?

1. Check line 100 (DEVICE configuration)
2. Look at line 668 (`_encode_batch` - FIX #4)
3. Verify line 2582 (model.to(device))

### Results don't make sense?

1. Check line 3046 (experiment config)
2. Look at line 1192 (run_experiment function)
3. Verify line 2652 (run_baseline_comparison)

---

## ✅ VALIDATION POINTS

**Key lines to verify file integrity:**

- ✅ Line 15: SECTION 1 header present
- ✅ Line 125: SECTION 2 header present
- ✅ Line 519: SECTION 3 header present
- ✅ Line 630: SECTION 4 header present
- ✅ Line 766: SECTION 5 header present
- ✅ Line 981: SECTION 6 header present
- ✅ Line 1189: SECTION 7 header present
- ✅ Line 1410: SECTION 8 header present
- ✅ Line 2350: SECTION 9 header present ← NEW
- ✅ Line 3025: SECTION 10 header present
- ✅ Line 3028: main() function present
- ✅ Line 3406: End of file marker present

---

**Last Updated:** 2025-12-10
**File Status:** ✅ Complete with 10 Sections
**Total Lines:** 3,406

Use this guide to navigate the code efficiently!
