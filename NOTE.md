# FL-bench Differential Privacy Extensions

---

## Table of Contents

- [Overview](#overview)
- [New Algorithms Implemented](#new-algorithms-implemented)
- [Algorithm Variants Explanation](#algorithm-variants-explanation)
- [Detailed File Changes](#detailed-file-changes)
- [New Directory Structure](#new-directory-structure)


---

## Overview

This document tracks all modifications made to FL-bench based on commit `bf956181573c2b18edd07f441cb7b46df6f1bb14`. The modifications introduce comprehensive differential privacy (DP) mechanisms for federated learning, including multiple algorithm implementations, privacy accounting tools, and experimental infrastructure.

### Summary Statistics

- **Total Files Changed:** 62
- **Lines Added:** 7,641
- **Lines Deleted:** 14
- **New Algorithms:** 5 (DP-FedAvg Local, DP-FedAvg Central, DP-SCAFFOLD, DP-FedStein, DP-ScaffStein)
- **Configuration Files:** 26 (15 main configs + 10 demo configs + 1 modified default)
- **Experiment Scripts:** 6
- **Demo Scripts:** 10

### Main Contributions

1. **DP-FedAvg Local** - Local differential privacy with per-sample gradient clipping
2. **DP-FedAvg Central** - Central differential privacy with server-side noise addition
3. **DP-SCAFFOLD** - SCAFFOLD algorithm enhanced with local differential privacy
4. **DP-FedStein** - DP-FedAvg integrated with James-Stein Estimator for noise reduction
5. **DP-ScaffStein** - DP-SCAFFOLD integrated with James-Stein Estimator

---

## New Algorithms Implemented

### 1. DP-FedAvg Local

**Description:** Implements local differential privacy using per-sample gradient computation via `torch.func.grad` and `vmap`. Each client adds noise to their gradients locally before sending updates to the server.

**Algorithm Variants:**
- `step_noise`: Add Gaussian noise to gradients at each training step
- `last_noise`: Add Gaussian noise to parameter differences after training completion

**Implementation Files:**
- `src/client/dp_fedavg_local.py` (469 lines)
- `src/server/dp_fedavg_local.py` (88 lines)

**Configuration Files:**
- `config/dp_fedavg_step_noise.yaml`
- `config/dp_fedavg_last_noise.yaml`

---

### 2. DP-FedAvg Central

**Description:** Implements central differential privacy where the server adds noise to aggregated updates. Clients perform standard training without noise.

**Implementation Files:**
- `src/client/dp_fedavg_central.py` (34 lines)
- `src/server/dp_fedavg_central.py` (94 lines)

---

### 3. DP-SCAFFOLD

**Description:** Combines the SCAFFOLD control variate mechanism with local differential privacy. Provides variance reduction through control variates while maintaining privacy guarantees.

**Algorithm Variants:**
- `step_noise`: Add noise at each gradient step with SCAFFOLD correction
- `last_noise`: Add noise to parameter differences with SCAFFOLD correction

**Implementation Files:**
- `src/client/dp_scaffold.py` (355 lines)
- `src/server/dp_scaffold.py` (113 lines)

**Configuration Files:**
- `config/dp_scaffold_step_noise.yaml`
- `config/dp_scaffold_last_noise.yaml`
- `config/dp_scaffold.yaml`

---

### 4. DP-FedStein

**Description:** Enhances DP-FedAvg Local with James-Stein Estimator (JSE) to reduce the impact of differential privacy noise while maintaining privacy guarantees. The JSE applies shrinkage to noisy updates, improving model accuracy.

**Algorithm Variants:**
- `last_noise_server_jse`: Add noise to parameter differences, apply JSE on server side (global shrinkage)
- `step_noise_step_jse`: Add noise at each step, apply JSE at each step (layerwise shrinkage)
- `step_noise_final_jse`: Add noise at each step, apply JSE to final aggregated result

**Implementation Files:**
- `src/client/dp_fed_stein.py` (507 lines)
- `src/server/dp_fed_stein.py` (384 lines)

**Configuration Files:**
- `config/dp_fedstein_last_noise_server_jse.yaml`
- `config/dp_fedstein_step_noise_step_jse.yaml`
- `config/dp_fedstein_step_noise_final_jse.yaml`

---

### 5. DP-ScaffStein

**Description:** Combines DP-SCAFFOLD with James-Stein Estimator, providing both variance reduction from SCAFFOLD control variates and noise reduction from JSE shrinkage.

**Algorithm Variants:**
- `last_noise_server_jse`: SCAFFOLD with parameter-level noise and server-side JSE
- `step_noise_step_jse`: SCAFFOLD with gradient-level noise and step-wise JSE
- `step_noise_final_jse`: SCAFFOLD with gradient-level noise and final JSE

**Implementation Files:**
- `src/client/dp_scaffstein.py` (353 lines)
- `src/server/dp_scaffstein.py` (317 lines)

**Configuration Files:**
- `config/dp_scaffstein_last_noise_server_jse.yaml`
- `config/dp_scaffstein_step_noise_step_jse.yaml`
- `config/dp_scaffstein_step_noise_final_jse.yaml`

---

## Algorithm Variants Explanation

### Noise Addition Strategies

**1. step_noise (Gradient-level Noise)**
- **When:** Noise is added to gradients at each training step
- **Formula:** σ_DP = C × σ_g / b
  - C: Clipping norm
  - σ_g: Noise parameter (sigma)
  - b: Batch size
- **Characteristics:** More frequent noise addition, noise scales with number of steps

**2. last_noise (Parameter-level Noise)**
- **When:** Noise is added to parameter differences after training completion
- **Formula:** σ_DP = C × K × η_l × σ_g / b
  - C: Clipping norm
  - K: Number of local epochs
  - η_l: Local learning rate
  - σ_g: Noise parameter (sigma)
  - b: Batch size
- **Characteristics:** Single noise addition, noise accumulates across all training steps

### JSE Application Methods

**1. step_jse (Step-wise JSE)**
- **Application:** Apply James-Stein shrinkage at each training step
- **Location:** Client-side, after gradient noise addition
- **Method:** Layerwise shrinkage to each parameter's gradient
- **Use Case:** DP-FedStein and DP-ScaffStein with `step_noise_step_jse` variant

**2. final_jse (Final JSE)**
- **Application:** Apply James-Stein shrinkage to final aggregated gradients
- **Location:** Client-side, after all training steps complete
- **Method:** Layerwise shrinkage to accumulated parameter differences
- **Use Case:** DP-FedStein and DP-ScaffStein with `step_noise_final_jse` variant

**3. server_jse (Server-side JSE)**
- **Application:** Apply James-Stein shrinkage on server to aggregated updates
- **Location:** Server-side, after aggregating client updates
- **Method:** Global shrinkage across all parameters
- **Use Case:** DP-FedStein and DP-ScaffStein with `last_noise_server_jse` variant

---

## Detailed File Changes

### Core DP Mechanisms (2 files)

**`src/utils/dp_mechanisms.py`** (126 lines, new file)
- **Functions:**
  - `add_gaussian_noise()`: Add calibrated Gaussian noise to tensors
  - `clip_gradients()`: Clip gradients by norm for DP
  - `compute_per_sample_grads()`: Compute per-sample gradients using `torch.func.grad` and `vmap`
  - `compute_per_sample_norms()`: Compute L2 norms of per-sample gradients
- **Purpose:** Core differential privacy utility functions for gradient clipping and noise addition

**`src/utils/jse_utils.py`** (396 lines, new file)
- **Class:** `JSEProcessor`
- **Methods:**
  - `apply_jse_shrinkage()`: Apply James-Stein shrinkage to a tensor
  - `apply_jse_shrinkage_to_mean()`: Apply mean-centered James-Stein shrinkage
  - `apply_layerwise_jse_to_gradients()`: Apply JSE to model gradients layer-by-layer
  - `apply_global_jse_to_gradients()`: Apply global JSE to all gradients with unified shrinkage
  - `apply_layerwise_jse_to_parameter_diff()`: Apply JSE to parameter differences layer-by-layer
  - `apply_global_jse_to_parameter_diff()`: Apply global JSE to parameter differences
- **Purpose:** James-Stein Estimator implementation for noise reduction with numerical stability

---

### Client Implementations (5 files)

1. **`src/client/dp_fedavg_central.py`** (34 lines, new file)
   - Central DP client with standard training (no local noise)

2. **`src/client/dp_fedavg_local.py`** (469 lines, new file)
   - Local DP client with per-sample gradient clipping and noise addition
   - Supports both step_noise and last_noise variants
   - Base class for DP-SCAFFOLD, DP-FedStein, and DP-ScaffStein

3. **`src/client/dp_scaffold.py`** (355 lines, new file)
   - Extends DPFedAvgLocalClient with SCAFFOLD control variates
   - Implements control variate corrections in gradient computation

4. **`src/client/dp_fed_stein.py`** (507 lines, new file)
   - Extends DPFedAvgLocalClient with James-Stein Estimator
   - Implements 3 JSE variants: last_noise_server_jse, step_noise_step_jse, step_noise_final_jse
   - Tracks shrinkage factors for analysis

5. **`src/client/dp_scaffstein.py`** (353 lines, new file)
   - Extends DPScaffoldClient with James-Stein Estimator
   - Combines SCAFFOLD control variates with JSE shrinkage

---

### Server Implementations (5 files)

1. **`src/server/dp_fedavg_central.py`** (94 lines, new file)
   - Central DP server with server-side noise addition to aggregated updates

2. **`src/server/dp_fedavg_local.py`** (88 lines, new file)
   - Local DP server with standard FedAvg aggregation
   - Supports configurable global learning rate

3. **`src/server/dp_scaffold.py`** (113 lines, new file)
   - DP-SCAFFOLD server managing control variates
   - Aggregates control variate updates from clients

4. **`src/server/dp_fed_stein.py`** (384 lines, new file)
   - DP-FedStein server with JSE application
   - Implements server-side JSE for last_noise_server_jse variant
   - Tracks and logs shrinkage factors

5. **`src/server/dp_scaffstein.py`** (317 lines, new file)
   - DP-ScaffStein server combining SCAFFOLD and JSE
   - Manages both control variates and JSE shrinkage

**Modified:**
- **`src/server/fedavg.py`** (+150 lines)
  - Enhanced base server class to support DP variants
  - Added return_diff mode support for parameter difference aggregation

---

### Configuration Files (26 files)

**Modified Configuration:**
- **`config/defaults.yaml`** (modified)
  - Updated default configurations to support DP algorithms

**New Main Configuration Files (15 files):**
- `config/defaults copy.yaml` (147 lines) - Backup of original defaults
- `config/dp_fedavg_last_noise.yaml` (39 lines)
- `config/dp_fedavg_step_noise.yaml` (39 lines)
- `config/dp_fedstein_last_noise_server_jse.yaml` (39 lines)
- `config/dp_fedstein_step_noise_final_jse.yaml` (39 lines)
- `config/dp_fedstein_step_noise_step_jse.yaml` (40 lines)
- `config/dp_scaffold.yaml` (78 lines)
- `config/dp_scaffold_last_noise.yaml` (43 lines)
- `config/dp_scaffold_step_noise.yaml` (45 lines)
- `config/dp_scaffstein_last_noise_server_jse.yaml` (41 lines)
- `config/dp_scaffstein_step_noise_final_jse.yaml` (46 lines)
- `config/dp_scaffstein_step_noise_step_jse.yaml` (42 lines)

**Demo Configuration Files (10 files in `demo/config/`):**
- `demo/config/defaults.yaml` (147 lines)
- `demo/config/dp_fedavg_last_noise.yaml` (39 lines)
- `demo/config/dp_fedavg_step_noise.yaml` (39 lines)
- `demo/config/dp_fedstein_last_noise_server_jse.yaml` (39 lines)
- `demo/config/dp_fedstein_step_noise_final_jse.yaml` (39 lines)
- `demo/config/dp_fedstein_step_noise_step_jse.yaml` (46 lines)
- `demo/config/dp_scaffold_last_noise.yaml` (43 lines)
- `demo/config/dp_scaffold_step_noise.yaml` (45 lines)
- `demo/config/dp_scaffstein_last_noise_server_jse.yaml` (41 lines)
- `demo/config/dp_scaffstein_step_noise_final_jse.yaml` (46 lines)
- `demo/config/dp_scaffstein_step_noise_step_jse.yaml` (42 lines)

---

### Data Generation Tools (2 files)

**Modified:**
- **`generate_data.py`** (minor modifications)
  - Updates to support DP experiments

**New:**
- **`generate_data_equal_samples.py`** (518 lines)
  - **Purpose:** Generate federated learning data partitions with equal samples per client
  - **Features:**
    - Ensures each client receives exactly the same number of samples
    - Similarity parameter controls data heterogeneity by mixing IID and non-IID samples
    - Supports multiple datasets (MNIST, CIFAR-10, etc.)
  - **Usage:** `python generate_data_equal_samples.py -d mnist -cn 100 -sim 0.5`

---

### Experiment Infrastructure (6 files)

1. **`experiment.py`** (84 lines, new file)
   - **Purpose:** Experiment orchestration script for running DP experiments
   - **Features:** Simplified experiment launching with Hydra configuration

2. **`run_test.sh`** (106 lines, new file)
   - **Purpose:** PBS batch experiment script
   - **Features:**
     - Automated job submission to PBS scheduler
     - Parameter sweeps over sigma, learning rate, epochs
     - Experiment metadata tracking
     - Job script generation

3. **`run_test_perlmutter.sh`** (125 lines, new file)
   - **Purpose:** Perlmutter supercomputer batch experiments
   - **Features:**
     - SLURM job scheduling
     - GPU resource allocation
     - Optimized for NERSC Perlmutter environment

4. **`run_test_perlmutter_bench.sh`** (124 lines, new file)
   - **Purpose:** Benchmark experiments on Perlmutter
   - **Features:** Performance benchmarking configurations

5. **`workstation_run_test.sh`** (98 lines, new file)
   - **Purpose:** Workstation batch experiments
   - **Features:** Local execution without job scheduler

6. **`test.sh`** (15 lines, new file)
   - **Purpose:** Quick test script for development
   - **Features:** Rapid testing of algorithm implementations

---

### Privacy Analysis Tools (2 files)

1. **`get_epsilon_bound.py`** (148 lines, new file)
   - **Purpose:** Privacy budget (epsilon) calculation using Rényi Differential Privacy (RDP)
   - **Features:**
     - RDP-based epsilon bound computation
     - Supports subsampling amplification
     - Composition analysis for multiple rounds
   - **Key Functions:**
     - `RDP_epsilon_bound_gaussian()`: Gaussian mechanism RDP bound
     - `cgf_subsampling_for_int_alpha()`: Subsampling CGF bound
     - Privacy accounting for FL with user and data subsampling

2. **`calculate_epsilon_example.py`** (135 lines, new file)
   - **Purpose:** Example epsilon calculator with parameter adjustment guidance
   - **Features:**
     - Interactive epsilon calculation examples
     - Demonstrates how to adjust FL parameters based on epsilon bounds
     - Parameter tuning guidance (T, K, M, sigma)
   - **Usage:** Configure meta-parameters and run to see privacy budget

---

### Visualization Tools (2 files)

1. **`csv/plot_accuracy.py`** (162 lines, new file)
   - **Purpose:** Test accuracy plotting tool for comparing multiple runs
   - **Features:**
     - Interactive file selection
     - Multi-run comparison plots
     - Customizable plot styling
   - **Usage:**
     ```bash
     python csv/plot_accuracy.py
     # Or programmatic: plot_test_accuracy(['file1.csv', 'file2.csv'])
     ```

2. **`csv/plot_shrinkage_factors.py`** (340 lines, new file)
   - **Purpose:** JSE shrinkage factor visualization
   - **Features:**
     - Shrinkage factor tracking across rounds
     - Statistical analysis of shrinkage behavior
     - Comparison across different JSE methods
   - **Usage:** Analyze shrinkage patterns from DP-FedStein/DP-ScaffStein experiments

---

### Demo Files (21 files)

**Documentation:**
- **`demo/README.md`** (86 lines, new file)
  - Chinese-language documentation for demo scripts
  - Usage instructions for all DP-FedAvg variants
  - Configuration parameter explanations

**Demo Main Scripts (10 files in `demo/`):**
- `demo/main_dp_fedavg_last_noise.py` (102 lines)
- `demo/main_dp_fedavg_step_noise.py` (102 lines)
- `demo/main_dp_fedstein_last_noise_server_jse.py` (112 lines)
- `demo/main_dp_fedstein_step_noise_final_jse.py` (111 lines)
- `demo/main_dp_fedstein_step_noise_step_jse.py` (110 lines)
- `demo/main_dp_scaffold_last_noise.py` (104 lines)
- `demo/main_dp_scaffold_step_noise.py` (104 lines)
- `demo/main_dp_scaffstein_last_noise_server_jse.py` (105 lines)
- `demo/main_dp_scaffstein_step_noise_final_jse.py` (105 lines)
- `demo/main_dp_scaffstein_step_noise_step_jse.py` (105 lines)

**Purpose:** Self-contained demonstration scripts for each algorithm variant with embedded configurations

---

### Model Updates (1 file)

**Modified:**
- **`src/utils/models.py`** (+46 lines)
  - **Purpose:** Model architecture additions to support DP training
  - **Changes:** Enhanced model initialization and parameter handling for DP compatibility

---

### Other Modifications (1 file)

**Modified:**
- **`.gitignore`** (8 modifications)
  - Updated to exclude experiment outputs, logs, and temporary files

---

## New Directory Structure

```
FL-bench/
├── config/
│   ├── defaults.yaml (modified)
│   ├── dp_fedavg_step_noise.yaml (new)
│   ├── dp_fedavg_last_noise.yaml (new)
│   ├── dp_scaffold_step_noise.yaml (new)
│   ├── dp_scaffold_last_noise.yaml (new)
│   ├── dp_scaffold.yaml (new)
│   ├── dp_fedstein_step_noise_step_jse.yaml (new)
│   ├── dp_fedstein_step_noise_final_jse.yaml (new)
│   ├── dp_fedstein_last_noise_server_jse.yaml (new)
│   ├── dp_scaffstein_step_noise_step_jse.yaml (new)
│   ├── dp_scaffstein_step_noise_final_jse.yaml (new)
│   └── dp_scaffstein_last_noise_server_jse.yaml (new)
│
├── src/
│   ├── client/
│   │   ├── dp_fedavg_central.py (new)
│   │   ├── dp_fedavg_local.py (new)
│   │   ├── dp_scaffold.py (new)
│   │   ├── dp_fed_stein.py (new)
│   │   └── dp_scaffstein.py (new)
│   │
│   ├── server/
│   │   ├── fedavg.py (modified, +150 lines)
│   │   ├── dp_fedavg_central.py (new)
│   │   ├── dp_fedavg_local.py (new)
│   │   ├── dp_scaffold.py (new)
│   │   ├── dp_fed_stein.py (new)
│   │   └── dp_scaffstein.py (new)
│   │
│   └── utils/
│       ├── models.py (modified, +46 lines)
│       ├── dp_mechanisms.py (new, 126 lines)
│       └── jse_utils.py (new, 396 lines)
│
├── demo/ (new directory)
│   ├── README.md (new, Chinese documentation)
│   ├── config/
│   │   ├── defaults.yaml (new)
│   │   ├── dp_fedavg_step_noise.yaml (new)
│   │   ├── dp_fedavg_last_noise.yaml (new)
│   │   └── ... (10 config files total)
│   │
│   ├── main_dp_fedavg_step_noise.py (new)
│   ├── main_dp_fedavg_last_noise.py (new)
│   ├── main_dp_fedstein_step_noise_step_jse.py (new)
│   ├── main_dp_fedstein_step_noise_final_jse.py (new)
│   ├── main_dp_fedstein_last_noise_server_jse.py (new)
│   ├── main_dp_scaffold_step_noise.py (new)
│   ├── main_dp_scaffold_last_noise.py (new)
│   ├── main_dp_scaffstein_step_noise_step_jse.py (new)
│   ├── main_dp_scaffstein_step_noise_final_jse.py (new)
│   └── main_dp_scaffstein_last_noise_server_jse.py (new)
│
├── csv/ (new directory)
│   ├── plot_accuracy.py (new, 162 lines)
│   └── plot_shrinkage_factors.py (new, 340 lines)
│
├── experiment.py (new, 84 lines)
├── generate_data_equal_samples.py (new, 518 lines)
├── get_epsilon_bound.py (new, 148 lines)
├── calculate_epsilon_example.py (new, 135 lines)
├── run_test.sh (new, 106 lines)
├── run_test_perlmutter.sh (new, 125 lines)
├── run_test_perlmutter_bench.sh (new, 124 lines)
├── workstation_run_test.sh (new, 98 lines)
├── test.sh (new, 15 lines)
└── NOTE.md (this file)
```


---

**End of NOTE.md**
