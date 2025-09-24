
- Privacy Protection and Statistical Efficiency Trade-Off for Federated Learning: 
    - ε ∈ {0:8, 2, 5}—and set δ to be 10^-5 the number of clients K � 15
    - ε ∈ {0:5, 0:8, 1}

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

- Differentially Private Federated Learning on Non-iid Data: Convergence Analysis and Adaptive Optimization
    - Parameter Setting: The overall number of clients n is set to
    100. Privacy level and failure probability are respectively set as
    ϵ = 0.3 and δ = 10−2. The default sampling ratio of clients is
    set as 0.1. We adjust the hyperparameters via grid search and
    give the optimal values as follows: over MNIST and FMNIST,
    we choose batch size B = 10, local iteration numbers τ = 300
    and local learning rate αl = 0.01; over SVHN and CIFAR10,
    the corresponding values are B = 50, τ = 50 and αl = 0.1,
    respectively. For Algorithm 2, we choose β1 = 0.9,β2 = 0.99
    and π = 10−3. We choose the global learning rate over MNIST
    and FMNIST as αg = 0.01and over SVHN and CIFAR10, this
    value is 0.005. Furthermore, we enable local and global learning
    rates to have the decaying rate 1/√t.

- Differentially private low-rank adaptation of large language model using federated learning
    - Wesettheprivacyparameters ϵ = 8 andδ= 10e−5.

- Differentially Private Federated Learning with Local Regularization and Sparsification
    - We set the privacy parameters ϵ = 8 
