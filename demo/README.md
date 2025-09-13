# DP-FedAvg Demo Collection

这个文件夹包含了DP-FedAvg算法的所有变体演示文件。

## 文件结构

```
demo/
├── configs/                          # 配置文件目录
│   ├── dp_fedavg_step_noise.yaml    # step_noise变体配置
│   └── dp_fedavg_last_noise.yaml    # last_noise变体配置
├── main_dp_fedavg_step_noise.py     # step_noise变体演示
├── main_dp_fedavg_last_noise.py     # last_noise变体演示
├── main_dp_local.py                 # 原始DP本地演示文件
└── README.md                        # 本说明文件
```

## 算法变体说明

### 1. Step Noise (梯度级噪声)
- **文件**: `main_dp_fedavg_step_noise.py`
- **配置**: `configs/dp_fedavg_step_noise.yaml`
- **特点**: 在每个训练步骤中向梯度添加噪声
- **噪声公式**: σ_DP = C × σ_g / b

### 2. Last Noise (参数级噪声)
- **文件**: `main_dp_fedavg_last_noise.py`
- **配置**: `configs/dp_fedavg_last_noise.yaml`
- **特点**: 训练完成后向参数差值添加噪声
- **噪声公式**: σ_DP = C × K × η_l × σ_g / b

## 使用方法

### 运行Step Noise变体
```bash
cd /Users/kfang11/Documents/personalcode/FL-bench
python demo/main_dp_fedavg_step_noise.py
```

### 运行Last Noise变体
```bash
cd /Users/kfang11/Documents/personalcode/FL-bench
python demo/main_dp_fedavg_last_noise.py
```

### 运行原始DP Local文件
```bash
cd /Users/kfang11/Documents/personalcode/FL-bench
python demo/main_dp_local.py
```

## 配置参数说明

每个配置文件都包含以下主要参数：

```yaml
dp_fedavg_local:
  algorithm_variant: "step_noise"  # 或 "last_noise"
  clip_norm: 1.0                   # 梯度裁剪阈值
  sigma: 0.1                       # 噪声标准差参数
```

## 自定义配置

您可以通过以下方式自定义参数：

1. **修改配置文件**: 直接编辑 `configs/` 中的YAML文件
2. **命令行覆盖**: 使用Hydra的参数覆盖功能

```bash
# 示例：修改噪声参数
python demo/main_dp_fedavg_step_noise.py dp_fedavg_local.sigma=0.2 dp_fedavg_local.clip_norm=2.0
```

## 输出结果

运行后，结果将保存在 `out/dp_fedavg_local/` 目录下，包括：
- 训练日志
- 学习曲线图
- 性能指标CSV文件

## 注意事项

1. 确保已安装所有依赖项
2. 首次运行需要下载数据集
3. 建议先用较少的轮数测试配置是否正确