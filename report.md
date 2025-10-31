# Federated Learning - Privacy Parameters Report

## Tried configurations
sigma_values=(0.003 0.03 0.3 3.0 30.0)

lr_values=(2.0 1.0 0.5)

global_epoch_values=(100)

local_epoch_values=(5 10 50)

clip_norm_values=(0.1 1.0 10.0 50.0 100.0)

data_sample_ratio_values=(0.1 0.2 0.5)

jse_way=("global" "per_layer")

## Default Configuration

**Parameters:**
- **T** (Communication Rounds): 100
- **K** (Local Updates): 50
- **M** (Number of Users): 60
- **R** (Training Data Points): 1000 (0.8 × 5000)
- **δ** (Privacy Parameter): 1/(M×R) ≈ 1.67×10⁻⁵
- **l** (User Subsampling Ratio): 0.2
- **s** (Data Subsampling Ratio): 0.2
- **σ_gaussian** (Noise Std Dev): 30.0



## Experimental Results

| Setting | epsilon (ε) | test accuracy (%) |
|-------------------|---------------------|---------------------|
| Default (σ=30.0, K=50, s=0.2) | 7.2627 | <20.0 |
| σ=30.0, K=5, s=0.2 | 6.0412 | <20.0 |
| σ=30.0, K=10, s=0.2 | 6.2737 | <20.0 |
| σ=3.0, K=5, s=0.2 | 6.3194 | 34.97 |
| σ=3.0, K=10, s=0.2 | 6.8269 | 24.53 |
| σ=3.0, K=50, s=0.2 | 9.4474 | 23.38 |
| σ=3.0, K=50, s=0.1 | 6.7973 | 20.98 |
| σ=3.0, K=50, s=0.5 | 20.0153 | 41.12 |
| σ=0.3, K=50, s=0.5 | 3839.9125 | 82.98 |
| σ=0.3, K=5, s=0.5 | 185.7343 | 78.37 |
| σ=0.3, K=5, s=0.2 | 29.3011 | 74.03 |
| σ=0.3, K=5, s=0.1 | 13.4033 | 66.68 |




