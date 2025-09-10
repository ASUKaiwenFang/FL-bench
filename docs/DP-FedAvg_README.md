# DP-FedAvg Implementation in FL-bench

This document provides comprehensive documentation for the Differential Privacy FedAvg (DP-FedAvg) implementations added to FL-bench.

## Overview

FL-bench now includes two separate differential privacy implementations for federated learning:

1. **DP-FedAvg-Local**: Local Differential Privacy implementation
2. **DP-FedAvg-Central**: Central Differential Privacy implementation

These implementations provide different privacy guarantees and are suitable for different trust models and use cases.

## Algorithms

### Local Differential Privacy (DP-FedAvg-Local)

**Trust Model**: Does not require trusting the server
**Privacy Mechanism**: Clients add calibrated noise to their gradients before sending updates
**Implementation Location**: Client-side noise addition

**Algorithm Steps**:
1. Client performs local training
2. **Gradient Clipping**: Clip gradients by L2 norm to bound sensitivity
3. **Noise Addition**: Add calibrated Gaussian noise to clipped gradients  
4. Apply noisy gradients to model parameters
5. Send noisy parameter updates to server
6. Server performs standard FedAvg aggregation

**Privacy Guarantees**: Each client's data satisfies (ε,δ)-differential privacy

### Central Differential Privacy (DP-FedAvg-Central)

**Trust Model**: Requires trusting the server to properly implement DP
**Privacy Mechanism**: Server adds noise to aggregated parameters
**Implementation Location**: Server-side noise addition after aggregation

**Algorithm Steps**:
1. Clients perform standard local training (no noise)
2. Clients send clean parameter updates to server
3. Server performs standard FedAvg aggregation
4. **Noise Addition**: Server adds calibrated noise to aggregated global parameters
5. Server distributes noisy global parameters to clients

**Privacy Guarantees**: The global aggregation process satisfies (ε,δ)-differential privacy

## Configuration

### Configuration File

Use the dedicated configuration file for DP-FedAvg:

```bash
# For Local DP
python main.py --config-name dp_fedavg method=dp_fedavg_local

# For Central DP  
python main.py --config-name dp_fedavg method=dp_fedavg_central
```

### Configuration Parameters

#### Local DP-FedAvg Parameters (`dp_fedavg_local`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 8.0 | Privacy budget parameter |
| `delta` | 1e-5 | Privacy parameter for (ε,δ)-DP |
| `clip_norm` | 1.0 | Gradient clipping norm |
| `noise_multiplier` | 1.1 | Noise multiplier for calibrated noise |
| `lot_size` | 64 | Logical batch size for privacy accounting |
| `accounting_mode` | "rdp" | Privacy accounting method [rdp, gdp, ma] |
| `adaptive_clipping` | false | Whether to use adaptive gradient clipping |
| `clip_percentile` | 50 | Percentile for adaptive clipping |

#### Central DP-FedAvg Parameters (`dp_fedavg_central`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 8.0 | Privacy budget parameter |
| `delta` | 1e-5 | Privacy parameter for (ε,δ)-DP |
| `server_noise_multiplier` | 0.8 | Server-side noise multiplier |
| `noise_type` | "gaussian" | Noise type [gaussian, laplace] |
| `sensitivity` | 1.0 | Sensitivity parameter for noise calibration |
| `clipping_mode` | "automatic" | Clipping mode [manual, automatic, adaptive] |
| `aggregation_weights` | "uniform" | Aggregation weights [uniform, adaptive] |

## Usage Examples

### Basic Usage

```bash
# Local DP with default parameters
python main.py --config-name dp_fedavg method=dp_fedavg_local

# Central DP with custom privacy budget
python main.py --config-name dp_fedavg method=dp_fedavg_central \
    dp_fedavg_central.epsilon=4.0 dp_fedavg_central.delta=1e-6

# Local DP with custom clipping and noise
python main.py --config-name dp_fedavg method=dp_fedavg_local \
    dp_fedavg_local.clip_norm=0.5 dp_fedavg_local.noise_multiplier=1.5
```

### Advanced Configuration

Create a custom configuration file (e.g., `config/my_dp_config.yaml`):

```yaml
defaults:
  - dp_fedavg

# Override specific parameters
method: dp_fedavg_local

dp_fedavg_local:
  epsilon: 4.0
  delta: 1e-6
  clip_norm: 0.8
  noise_multiplier: 1.3
  lot_size: 32

# Modify common settings
common:
  global_epoch: 50
  local_epoch: 3
  batch_size: 16
  
dataset:
  name: cifar10
  
model:
  name: cnn
```

## Privacy Analysis

### Privacy Budget Management

Both implementations include privacy accounting mechanisms:

**Local DP**:
- Each client tracks its own privacy consumption
- Privacy budget is distributed across training rounds
- Clients stop participating when budget is exhausted

**Central DP**:
- Server tracks global privacy consumption
- Privacy cost accumulates with each aggregation round
- Global budget shared across all participants

### Privacy Parameters Selection

**Epsilon (ε)**: Privacy budget
- Smaller ε → Better privacy, worse utility
- Typical values: 1.0 (strong privacy) to 10.0 (weaker privacy)
- Recommendation: Start with ε=8.0 for experimentation

**Delta (δ)**: Failure probability
- Should be much smaller than 1/dataset_size
- Typical values: 1e-5 to 1e-6
- Recommendation: Use δ=1e-5 for most cases

**Noise Multiplier**: Controls noise magnitude
- Higher values → Better privacy, worse utility  
- Automatically computed based on ε, δ if not specified
- Manual tuning may be needed for optimal performance

### Privacy-Utility Tradeoffs

| Configuration | Privacy | Utility | Use Case |
|---------------|---------|---------|----------|
| ε=1.0, clip=0.5, noise=2.0 | Very High | Low | High privacy requirements |
| ε=4.0, clip=1.0, noise=1.5 | High | Medium | Balanced privacy-utility |
| ε=8.0, clip=1.0, noise=1.1 | Medium | High | Research/benchmarking |
| ε=16.0, clip=2.0, noise=0.8 | Low | Very High | Weak privacy baseline |

## Implementation Details

### File Structure

```
FL-bench/
├── src/
│   ├── client/
│   │   ├── dp_fedavg_local.py      # Local DP client implementation
│   │   └── dp_fedavg_central.py    # Central DP client implementation  
│   ├── server/
│   │   ├── dp_fedavg_local.py      # Local DP server implementation
│   │   └── dp_fedavg_central.py    # Central DP server implementation
│   └── utils/
│       └── dp_mechanisms.py        # DP utility functions
├── config/
│   └── dp_fedavg.yaml              # DP-specific configuration
├── tests/
│   ├── test_dp_mechanisms.py       # Unit tests for DP mechanisms
│   └── test_dp_fedavg_integration.py # Integration tests
└── docs/
    └── DP-FedAvg_README.md         # This documentation
```

### Key Components

#### DP Mechanisms (`src/utils/dp_mechanisms.py`)

- `add_gaussian_noise()`: Add calibrated Gaussian noise
- `add_laplace_noise()`: Add Laplace noise for ε-DP
- `clip_gradients()`: Gradient clipping by L2 norm
- `privacy_accountant()`: Privacy cost calculation
- `PrivacyEngine`: Privacy consumption tracking

#### Client Implementations

**Local DP Client**: Implements gradient clipping and noise addition in `fit()` method
**Central DP Client**: Standard training with privacy tracking for server-side accounting

#### Server Implementations

**Local DP Server**: Standard aggregation with privacy consumption tracking
**Central DP Server**: Noise addition after aggregation in `aggregate_client_updates()`

## Testing

### Unit Tests

Run differential privacy mechanism tests:

```bash
cd FL-bench
python -m pytest tests/test_dp_mechanisms.py -v
```

### Integration Tests  

Run full algorithm integration tests:

```bash
python -m pytest tests/test_dp_fedavg_integration.py -v
```

### End-to-End Testing

Test complete training pipeline:

```bash
# Local DP end-to-end test
python main.py --config-name dp_fedavg method=dp_fedavg_local \
    common.global_epoch=5 common.local_epoch=1

# Central DP end-to-end test  
python main.py --config-name dp_fedavg method=dp_fedavg_central \
    common.global_epoch=5 common.local_epoch=1
```

## Performance Considerations

### Computational Overhead

**Local DP**:
- Additional gradient clipping computation
- Noise generation for each parameter update
- Privacy accounting calculations
- Overhead: ~10-20% increase in training time

**Central DP**:
- Server-side noise generation
- Privacy accounting at server
- Minimal client-side overhead
- Overhead: ~5-10% increase in training time

### Memory Usage

- Both implementations have minimal memory overhead
- Privacy engines maintain small state (~1KB per client)
- Noise generation is done in-place when possible

### Communication

- Local DP: Same communication cost as standard FedAvg
- Central DP: Same communication cost as standard FedAvg
- Privacy metadata adds negligible communication overhead

## Best Practices

### Parameter Selection

1. **Start with recommended defaults** and adjust based on privacy-utility requirements
2. **Use privacy accounting tools** to understand actual privacy consumption
3. **Test multiple configurations** to find optimal privacy-utility tradeoff
4. **Consider dataset size** when setting δ parameter

### Privacy Budget Management

1. **Set overall privacy budget** before training begins
2. **Monitor privacy consumption** throughout training
3. **Stop training** when budget is exhausted (Local DP)
4. **Use composition theorems** for multiple training runs

### Debugging and Monitoring

1. **Enable privacy logging** with `common.save_privacy_log: true`
2. **Check privacy consumption** regularly during training
3. **Validate noise addition** with unit tests
4. **Monitor model convergence** to ensure utility

## Troubleshooting

### Common Issues

**Poor Convergence**:
- Reduce noise multiplier or increase clipping norm
- Increase number of local epochs
- Use learning rate scheduling

**Privacy Budget Exhausted Too Quickly**:
- Increase epsilon or reduce number of training rounds
- Use adaptive privacy budget allocation
- Consider stronger composition theorems

**Memory Issues**:
- Reduce batch size if using large models
- Use gradient accumulation for large batches
- Enable memory optimization flags

### Error Messages

**"Privacy budget exhausted"**: Client has consumed allocated privacy budget
**"Invalid noise multiplier"**: Check that noise_multiplier > 0
**"Gradient clipping failed"**: Ensure clip_norm > 0 and gradients exist

## References

1. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
2. **DP-SGD**: Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)  
3. **Local DP**: Duchi et al., "Local Privacy and Statistical Minimax Rates" (FOCS 2013)
4. **Central DP**: Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis" (TCC 2006)

## Contributing

To contribute improvements to the DP-FedAvg implementation:

1. Fork the FL-bench repository
2. Create a feature branch for your changes
3. Add comprehensive tests for new functionality  
4. Update documentation as needed
5. Submit a pull request with detailed description

For questions or issues specific to DP-FedAvg, please open an issue with the "differential-privacy" label.
