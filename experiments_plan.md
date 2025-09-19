## Structured Experiment Plan for Federated Learning with Differential Privacy

### 1. Introduction and Research Questions

**Objective**: To systematically investigate the performance of various differentially private federated learning algorithms under different conditions of data heterogeneity.

**Core Research Questions**:
- How do algorithms like `dp_scaffstein`, `dp_fed_stein`, `dp_scaffold`, and `dp_fedavg` perform under varying degrees of data heterogeneity (non-IID) and differential privacy constraints?

**Key Hypotheses**:
1.  **Hypothesis 1 (Client Drift)**: In highly non-IID settings (e.g., low Dirichlet alpha value), `dp_scaffstein` is expected to achieve significantly higher model accuracy compared to `dp_scaffold` and `dp_fedavg` due to its explicit mechanism for correcting client drift.
2.  **Hypothesis 2 (SVGD Component)**: The Stein Variational Gradient Descent (SVGD) component within `dp_scaffstein` and `dp_fed_stein` will lead to faster convergence and superior final performance in highly heterogeneous environments compared to their non-Stein counterparts.

### 2. Experimental Design

#### 2.1 Key Parameters
-   **Dataset**: **MNIST**, EMNIST, (FEMNIST, CIFAR10)
-   **Data Distribution**: Dirichlet with α ∈ {0.0, 0.1, 1.0}
-   **Model Architecture**: **CNN**, 2NN
-   **Methods**: **dp_scaffstein**, dp_fed_stein, dp_scaffold, dp_fedavg
-   **Algorithm Variants**: step_noise_step_jse, step_noise_final_jse, last_noise_server_jse
-   **Client Configuration**: (Total Clients, Sampled Clients) ∈ {(10, 10), **(100, 10)**, (100, 20)}
-   **Training Hyperparameters**:
    -   `local_epochs`: 5, 10
    -   `global_rounds`: 50, 100
    -   `batch_size`: 32, 64
-   **Privacy Parameters**:
    -   `sigma` (DP noise multiplier): 0.1, 0.5, 1.0

#### 2.2 Baseline Configuration
To ensure a consistent basis for comparison, the following baseline configuration will be used:
-   `dataset`: **MNIST**
-   `data_distribution`: **Dirichlet (α=0.1)** (represents a high degree of non-IID)
-   `model`: **CNN**
-   `clients`: **(100 total, 10 sampled)**
-   `local_epochs`: **5**
-   `global_rounds`: **50**
-   `batch_size`: **32**
-   `sigma`: **0.5**

#### 2.3 Comparative Analyses
Experiments will be structured around the baseline to isolate variables:
1.  **Analysis 1: Impact of Data Heterogeneity**
    -   **Objective**: Evaluate algorithm robustness to data distribution shifts.
    -   **Procedure**: Using the baseline configuration, run `dp_scaffstein` and `dp_scaffold` while varying the `Dirichlet` parameter α across {0.0, 0.1, 1.0}.
2.  **Analysis 2: Impact of Privacy Budget**
    -   **Objective**: Plot the privacy-utility trade-off curve for key algorithms.
    -   **Procedure**: Using the baseline, vary the `sigma` parameter across {0.1, 0.5, 1.0}.
3.  **Analysis 3: Algorithm Variant Comparison**
    -   **Objective**: Determine the most effective noise injection strategy for `dp_scaffstein`.
    -   **Procedure**: Compare the performance of `step_noise_step_jse`, `step_noise_final_jse`, and `last_noise_server_jse` under the baseline configuration.

### 3. Evaluation Metrics

-   **Primary Metrics**:
    -   `Test Accuracy (Top-1)`: Accuracy of the final global model on the centralized test set.
    -   `Test Loss`: Corresponding loss value.
-   **Secondary Metrics**:
    -   `Convergence Rate`: Number of communication rounds required to reach a target accuracy (e.g., 80%).
    -   `Fairness`: The standard deviation of model accuracy across all clients, measuring performance consistency.

### 4. Experiment Tracking

-   **Output Directory**: The existing structure `out/{method}/{dataset}/{timestamp}/` will be maintained.
-   **Naming Convention**: A descriptive tag will be used for each run to easily identify its configuration, e.g., `mnist_cnn_dpschaffstein_dir0.1_sigma0.5`.

### 5. Phases of Experimentation

-   **Phase 1: Baseline Establishment and Hypothesis Validation**
    -   **Goal**: Validate the primary hypothesis that standard DP-FL algorithms (`dp-fedavg`, `dp-scaffold`) underperform in highly non-IID settings where `dp_scaffstein` is expected to excel.
    -   **Action**: Execute runs for `dp_fedavg`, `dp_scaffold`, and `dp_scaffstein` using the defined baseline configuration and compare initial results.









