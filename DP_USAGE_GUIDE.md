# Differential Privacy Parameter Adjustment Guide

## 1. Understanding epsilon (ε) values

**Lower ε value = Stronger privacy protection = Potentially worse model performance**

- **ε < 1**: Extremely strong privacy protection
- **1 ≤ ε < 5**: Strong privacy protection
- **5 ≤ ε < 10**: Moderate privacy protection
- **ε ≥ 10**: Weaker privacy protection

## 2. Usage Steps

### Step 1: Calculate ε value for current configuration
```bash
# Modify parameters in calculate_epsilon_example.py to match your configuration
python calculate_epsilon_example.py
```

### Step 2: Choose configuration based on privacy requirements

#### High privacy requirements (ε ≈ 3.56)
```bash
python main_dp_local.py --config-path config_examples --config-name high_privacy_dp_fedavg
```

#### Balanced requirements (ε ≈ 4.79)
```bash
python main_dp_local.py --config-path config_examples --config-name balanced_dp_fedavg
```

#### Performance priority (ε ≈ 8.73)
```bash
python main_dp_local.py --config-path config_examples --config-name performance_dp_fedavg
```

## 3. Parameter Adjustment Reference Table

| sigma value | epsilon value | Privacy level | Use case |
|-------------|---------------|---------------|----------|
| 5.0         | ~3.56         | Extremely strong | Medical data, financial data |
| 1.0         | ~4.79         | Strong        | General sensitive data |
| 0.5         | ~8.73         | Moderate      | Performance-critical scenarios |

## 4. Practical Usage Examples

### Modify existing configuration
If you have existing configuration files, just modify the `sigma` parameter:

```yaml
dp_fedavg_local:
  algorithm_variant: "step_noise"
  clip_norm: 1.0
  sigma: 5  # Change from 1 to 5, increase privacy protection
```

### Command line override
You can also override parameters directly from command line:
```bash
python main_dp_local.py method=dp_fedavg_local dp_fedavg_local.sigma=5
```

## 5. Important Notes

1. **Noise vs Performance Trade-off**: Higher sigma = stronger privacy protection, but slower model convergence
2. **Learning rate adjustment**: When increasing noise, you may need to slightly increase learning rate
3. **Dataset size impact**: Smaller datasets provide stronger privacy protection for the same sigma value
4. **Experimental validation**: Test different sigma values on actual data to find optimal settings

## 6. Quick Decision Guide

**How sensitive is my data?**
- Medical/financial data → Use sigma=5 (ε~3.56)
- Personal information data → Use sigma=1 (ε~4.79)
- Public or semi-public data → Use sigma=0.5 (ε~8.73)

**What are my model performance requirements?**
- Must have high accuracy → Use smaller sigma
- Can accept accuracy loss → Use larger sigma
- Need fast convergence → Use smaller sigma