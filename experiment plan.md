# Experiments Plan

## 1. Differentially Private Federated Learning: A Systematic Review

This systematic literature review identifies four relevant local differential privacy federated learning (DP-FL) papers:
- Private, Efficient, and Accurate: Protecting Models Trained by Multi-party Learning with Differential Privacy
- ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations
- Adaptive DP-FL: Differentially Private Federated Learning with Adaptive Noise
- Differentially Private Federated Learning on Heterogeneous Data

The first three papers provide limited experimental settings directly applicable to our research objectives. However, they establish a benchmark by commonly utilizing MNIST, FashionMNIST, and CIFAR-10 datasets, which will be incorporated into our experimental design.

The fourth paper, *Differentially Private Federated Learning on Heterogeneous Data*, provides the primary baseline configuration for our experiments with the following settings:
- Dataset: MNIST
- Model: NN1 (2-layer MLP)
- #Clients: 60
- #Global Rounds: 100
- #Local Rounds: 50
- Client Sample Rate: 0.2
- Data Sample Rate: 0.2
- Sigma: 30
- (ε, δ): (7.2, 1e-5)
- η: not specified in the original paper

Several experimental parameters require systematic investigation:
- **Local Learning Rate**: Initial exploration across {0.1, 0.05, 0.01, 0.005, 0.001}, followed by fine-grained grid search (10 points) within the optimal range identified from preliminary results.
- **Model Architecture**: The original 2-layer MLP baseline will be extended to include deeper neural networks (2NN) and convolutional neural networks (CNN). Note that optimal learning rates may vary across different model architectures.
- **Fixed Parameters**: The following parameters will remain constant to maintain the target privacy level (ε, δ): number of clients, global rounds, local rounds, client sampling rate, data sampling rate, and noise multiplier (σ). These settings align with the baseline paper's privacy guarantee.
- **Datasets**: MNIST will serve as the initial testbed, followed by FashionMNIST and CIFAR-10.

### Experimental Protocol for DP-SCAFFSTEIN

The experimental protocol consists of three phases:

1. **Primary Experiment**: Conduct DP-SCAFFSTEIN experiments following the aforementioned settings. The local learning rate will undergo two-stage tuning: (i) coarse-grained search over the range [0.1, 1.0], and (ii) fine-grained grid search (10 points) between the two best-performing values from phase (i).

2. **Baseline Comparisons**: Two baseline methods are essential for comprehensive evaluation:
   - **DP-FEDAVG**: Serves as the fundamental baseline for all differential privacy federated learning approaches and specifically for evaluating control variate techniques.
   - **DP-FEDSTEIN**: Provides a baseline for assessing the James-Stein Estimator (JSE) component independently.

   This dual-baseline approach enables isolation of performance contributions: comparing DP-SCAFFSTEIN with DP-FEDSTEIN reveals the impact of JSE, while comparison with DP-FEDAVG demonstrates overall improvement.

3. **MNIST Optimization**: Priority will be given to achieving superior performance on MNIST compared to the original DP-SCAFFOLD method. This focus is justified by MNIST's ubiquity in DP-FL literature as a standard benchmark. Furthermore, preliminary experiments on MNIST will provide insights into JSE performance characteristics, as model architecture selection significantly influences JSE effectiveness.



## Experimental Extensions

Upon demonstrating improved performance of DP-SCAFFSTEIN over DP-SCAFFOLD on MNIST, the experimental scope will expand systematically:

### Phase 1: Cross-Dataset Validation
Apply the optimized MNIST configuration to FashionMNIST and CIFAR-10 to assess generalizability across datasets with similar settings.

### Phase 2: Robustness Analysis

**1. Data Partitioning Strategies**

The baseline paper employs similarity-based metrics ensuring equal data distribution across clients. Alternative partitioning schemes warrant investigation, particularly Dirichlet distribution-based allocation, which may better represent real-world heterogeneity. Preliminary observations suggest that data partitioning significantly impacts performance when individual clients receive smaller data volumes, potentially due to batch size constraints relative to local dataset size. This may explain the baseline paper's choice of 60 clients rather than 100. Testing DP-SCAFFSTEIN under varied partitioning schemes will establish its robustness to data heterogeneity. Dirichlet-based partitioning is particularly relevant as it is more commonly adopted in recent federated learning literature.

**2. Privacy Budget Variations**

Privacy guarantees are determined by the interplay of multiple hyperparameters: number of clients, global rounds, local rounds, client sampling rate, data sampling rate, and noise multiplier (σ). The baseline paper's configuration of 50 local epochs appears notably high for DP settings. Exploring alternative privacy budget allocations is essential to characterize whether DP-SCAFFSTEIN provides consistent improvements across all privacy regimes or demonstrates advantages in specific configurations. The following reference provides guidance for alternative experimental settings:

Reference:
- PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation
    | Dataset | DP | Architecture | # Clients | # Rounds | Sample Rate | (ε, δ) | η | Batch Size |
    |---------|----|----|---------|--------|-------|--------|---|--------|
    | MNIST | LDP/CDP | 3-layer DNN | 100 | 150 | 1 | (8, 1e-3) | 1e-1/5e-3 | 64 |
    | Fashion-MNIST | LDP/CDP | 3-layer DNN | 100 | 150 | 0.3 | (8, 1e-3) | 1e-1/5e-3 | 64 |
    | EMNIST | LDP/CDP | 3-layer DNN | 100 | 100 | 0.3 | (8, 1e-3) | 1e-1 | 64 |
    | CH-MNIST | LDP/CDP | AlexNet | 40 | 30/150 | 0.8 | (8, 1e-3) | 1e-1/1e-4 | 64 |
    | CIFAR-10 | LDP/CDP | ResNet | 100 | 150/60 | 1 | (8, 1e-3) | 1e-1/5e-4 | 64/16 |
    | CIFAR-100 | LDP/CDP | CLIP + 1-layer DNN | 100 | 20 | 1 | (8, 1e-3) | 1e-1 | 4/250 |
    | Purchase-100 | LDP/CDP | 4-layer DNN | 50 | 580/150 | 0.1 | (8, 1e-3) | 1e-1/5e-3 | 64 |


## Exploratory Research Questions

The following research questions emerge from preliminary observations and warrant systematic investigation:

1. **Gradient Clipping Strategies**: Comparative analysis of three approaches:
   - Flat clipping with fixed maximum norm
   - Per-layer clipping with fixed maximum norm
   - Per-layer clipping with heuristic median-based adaptive norm

2. **JSE Granularity**: Investigation of flat versus per-layer James-Stein Estimator application. Preliminary observations indicate substantial variation in shrinkage multipliers across layers, with some layers exhibiting negative or near-zero values, leading to minimal clipping bounds (e.g., 0.01). The adequacy of such small clipping bounds requires empirical validation.

3. **Global Learning Rate Sensitivity**: Alternative literature suggests global learning rates as high as 1.0. A systematic exploration of global learning rate impact on DP-SCAFFSTEIN performance is necessary to establish optimal training dynamics. 