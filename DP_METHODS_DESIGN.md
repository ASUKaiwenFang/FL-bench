# FL-bench差分隐私方法架构设计文档

## 1. 概述

本文档详细记录了FL-bench框架中四种差分隐私联邦学习方法的分阶段实现设计：
- **DPFedAvg**: 基础差分隐私联邦平均算法
- **DPFedStein**: 差分隐私联邦学习 + James-Stein估计器
- **DPScaffold**: 差分隐私 + Scaffold控制变量方法
- **DPScaffoldStein**: 完整组合版本（DP + Scaffold + JSE）

每种方法支持三种算法变体，对应不同的噪声添加位置和JSE应用时机。

## 2. 现有架构分析

### 2.1 已有实现分析

#### DPFedAvgLocalClient (已存在)
- **文件**: `src/client/dp_fedavg_local.py`
- **特点**: 使用Opacus GradSampleModule实现per-sample梯度计算
- **噪声位置**: 梯度级别（每个训练步骤）
- **噪声公式**: $\sigma_{DP} = \mathcal{C} \sigma_g / b$
- **架构**: 继承自`FedAvgClient`
- **关键方法**:
  - `_compute_per_sample_norms_opacus()`: 计算每个样本的梯度范数
  - `_clip_and_add_noise_opacus()`: 梯度裁剪和噪声添加

#### SCAFFOLDClient (已存在)
- **文件**: `src/client/scaffold.py`
- **特点**: 实现Scaffold控制变量机制
- **架构**: 继承自`FedAvgClient`
- **关键组件**:
  - `c_local`: 本地控制变量
  - `c_global`: 全局控制变量
  - 控制变量修正：`param.grad.data += (c - c_i).to(self.device)`

### 2.2 架构模式观察
- **继承关系**: 所有客户端类继承自`FedAvgClient`
- **配对设计**: 每个客户端类对应一个服务器类
- **配置驱动**: 使用Hydra配置系统管理参数

## 3. 三种算法变体定义

## DPFedAvg的两种噪声策略

对于**DPFedAvg**（不包含JSE），只需要两种基本的噪声添加策略：

### last_noise: 参数级噪声添加
- **噪声位置**: 添加到参数差值 $(y_{i,K}^r - x^{r-1})$
- **噪声时机**: 本地训练完成后
- **噪声标准差**: $\sigma_{DP} = \mathcal{C} K \eta_\ell \sigma_g / b$

### step_noise: 梯度级噪声添加
- **噪声位置**: 添加到每步的梯度
- **噪声时机**: 每个训练步骤
- **噪声标准差**: $\sigma_{DP} = \mathcal{C} \sigma_g / b$

## DPFedStein的三种JSE策略

对于**DPFedStein**（包含JSE），基于上述两种噪声策略，扩展为三种JSE应用策略：

### last_noise_server_jse: 参数级噪声 + 服务器端JSE
- **对应**: 论文算法1
- **噪声策略**: 使用last_noise
- **JSE位置**: 服务器端聚合时

### step_noise_step_jse: 梯度级噪声 + 每步JSE
- **对应**: 论文算法2
- **噪声策略**: 使用step_noise
- **JSE位置**: 每步梯度噪声后立即应用

### step_noise_final_jse: 梯度级噪声 + 最终JSE
- **对应**: 论文算法3
- **噪声策略**: 使用step_noise
- **JSE位置**: 本地训练完成后，对参数差值应用

## 4. 分阶段实现计划

### 阶段1: 扩展DPFedAvg支持两种噪声策略

#### 4.1 修改策略
扩展现有的`DPFedAvgLocalClient`以支持两种噪声策略，不包含JSE功能：

```python
class DPFedAvgLocalClient(FedAvgClient):
    # 定义算法变体常量
    ALGORITHM_VARIANTS = {
        'last_noise': 1,
        'step_noise': 2
    }
    
    def __init__(self, **commons):
        super().__init__(**commons)
        
        # 支持字符串或数字配置
        variant_config = getattr(self.args.dp_fedavg_local, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
            
        # 保持现有参数
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        
    def fit(self):
        if self.algorithm_variant == 1:  # last_noise
            return self._last_noise_training()
        elif self.algorithm_variant == 2:  # step_noise
            return self._step_noise_training()
```

#### 4.2 关键实现点

**last_noise实现**:
```python
def _last_noise_training(self):
    """参数级噪声添加"""
    # 标准训练，不添加梯度噪声
    for k in range(self.local_epoch):
        x, y = self.get_data_batch()
        self._standard_training_step(x, y)  # 无噪声训练
    
    # 训练完成后添加参数级噪声
    param_diff = self._get_parameter_difference()
    noisy_param_diff = self._add_parameter_level_noise(param_diff)
    
    return noisy_param_diff
```

**step_noise实现（基于现有）**:
```python
def _step_noise_training(self):
    """梯度级噪声添加"""
    for k in range(self.local_epoch):
        x, y = self.get_data_batch()
        
        # 现有的梯度噪声逻辑
        self._compute_per_sample_gradients(x, y)
        self._clip_and_add_noise_opacus()
        self.optimizer.step()
    
    # 返回参数差值
    return self._get_parameter_difference()
```

### 阶段2: JSE工具模块

#### 2.1 模块设计
```python
# src/utils/jse_utils.py
import torch
from typing import Union, List

class JSEProcessor:
    """James-Stein估计器处理器"""
    
    @staticmethod
    def apply_jse_shrinkage(
        tensor: torch.Tensor, 
        noise_variance: float, 
        k_factor: int = 1
    ) -> torch.Tensor:
        """
        对张量应用James-Stein收缩
        
        Args:
            tensor: 待处理的张量
            noise_variance: 噪声方差 σ²
            k_factor: 累积因子（多步累积时使用）
            
        Returns:
            收缩后的张量
        """
        d = tensor.numel()  # 有效维度
        tensor_norm_sq = tensor.norm().item() ** 2
        
        if tensor_norm_sq > 1e-8:  # 数值稳定性检查
            shrinkage_factor = (d - 2) * k_factor * noise_variance / tensor_norm_sq
            shrinkage_factor = max(0.0, 1.0 - shrinkage_factor)
            return shrinkage_factor * tensor
        else:
            return tensor
    
    @staticmethod
    def apply_jse_to_gradients(
        model_parameters: List[torch.Tensor], 
        noise_variance: float
    ) -> None:
        """对模型梯度应用JSE（就地修改）"""
        for param in model_parameters:
            if param.grad is not None:
                param.grad.data = JSEProcessor.apply_jse_shrinkage(
                    param.grad.data, noise_variance
                )
    
    @staticmethod
    def apply_jse_to_parameter_diff(
        param_diff: torch.Tensor, 
        noise_variance: float, 
        k_factor: int = 1
    ) -> torch.Tensor:
        """对参数差值应用JSE"""
        return JSEProcessor.apply_jse_shrinkage(param_diff, noise_variance, k_factor)
    
    @staticmethod
    def compute_noise_variance(
        clip_norm: float, 
        sigma: float, 
        batch_size: int, 
        k_factor: int = 1
    ) -> float:
        """计算噪声方差"""
        sigma_dp = clip_norm * sigma / batch_size
        return (sigma_dp ** 2) * k_factor
```

#### 2.2 集成策略
- **客户端集成**: 在训练循环中调用JSE处理
- **服务器端集成**: 在聚合阶段调用JSE处理
- **配置控制**: 通过`jse_enabled`参数控制是否启用

### 阶段3: DPFedStein实现

#### 3.1 继承设计
```python
# src/client/dp_fed_stein.py
from src.client.dp_fedavg_local import DPFedAvgLocalClient

class DPFedSteinClient(DPFedAvgLocalClient):
    """DPFedAvg + JSE的完整实现"""
    
    def __init__(self, **commons):
        super().__init__(**commons)
        # 覆盖配置，默认启用JSE
        self.jse_enabled = True
        
        # 可选择算法变体，默认为step_noise_step_jse
        variant_config = getattr(
            self.args.dp_fed_stein, 'algorithm_variant', 'step_noise_step_jse'
        )
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
```

#### 3.2 服务器端实现
```python
# src/server/dp_fed_stein.py
from src.server.dp_fedavg_local import DPFedAvgLocalServer
from src.utils.jse_utils import JSEProcessor

class DPFedSteinServer(DPFedAvgLocalServer):
    client_cls = DPFedSteinClient
    
    def aggregate_client_updates(self, client_packages):
        # 如果是变体1，在服务器端应用JSE
        if self._is_variant_1():
            self._apply_server_side_jse(client_packages)
        
        # 调用父类的聚合逻辑
        super().aggregate_client_updates(client_packages)
```

### 阶段4: DPScaffold实现

#### 4.1 基础架构
```python
# src/client/dp_scaffold.py
from src.client.scaffold import SCAFFOLDClient
from opacus import GradSampleModule
from src.utils.jse_utils import JSEProcessor

class DPScaffoldClient(SCAFFOLDClient):
    """Scaffold + DP的组合实现"""
    
    # 复用算法变体常量
    ALGORITHM_VARIANTS = {
        'last_noise_server_jse': 1,
        'step_noise_step_jse': 2, 
        'step_noise_final_jse': 3
    }
    
    def __init__(self, **commons):
        super().__init__(**commons)
        
        # 添加DP相关参数
        self.clip_norm = self.args.dp_scaffold.clip_norm
        self.sigma = self.args.dp_scaffold.sigma
        
        # 支持字符串或数字配置
        variant_config = getattr(self.args.dp_scaffold, 'algorithm_variant', 'step_noise_step_jse')
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
            
        self.jse_enabled = getattr(self.args.dp_scaffold, 'jse_enabled', False)
        
    def set_parameters(self, package):
        super().set_parameters(package)
        # 复用DPFedAvg的模型包装逻辑
        self._setup_dp_model()
    
    def _setup_dp_model(self):
        """设置DP模型包装"""
        if not isinstance(self.model, GradSampleModule):
            original_device = self.model.device
            self.model = GradSampleModule(self.model)
            self.model.device = original_device
            # 更新参数名称以兼容Scaffold
            self._update_scaffold_param_names()
```

#### 4.2 关键挑战与解决方案

**挑战1: 控制变量与DP噪声的交互**
```python
def fit(self):
    """Scaffold + DP的训练循环"""
    for k in range(self.local_epoch):
        x, y = self.get_data_batch()
        
        # 计算per-sample梯度
        self._compute_per_sample_gradients(x, y)
        
        # DP处理: 裁剪和噪声
        if self.algorithm_variant in [2, 3]:
            self._clip_and_add_noise_opacus()
        
        # JSE处理（变体2）
        if self.algorithm_variant == 2 and self.jse_enabled:
            self._apply_jse_to_current_gradients()
        
        # Scaffold控制变量修正
        self._apply_scaffold_correction()
        
        # 参数更新
        self.optimizer.step()
    
    # 变体1的参数级噪声处理
    if self.algorithm_variant == 1:
        return self._handle_variant_1_post_training()
    
    # 变体3的最后JSE处理
    if self.algorithm_variant == 3 and self.jse_enabled:
        return self._handle_variant_3_final_jse()
```

**挑战2: 参数名称兼容性**
```python
def _update_scaffold_param_names(self):
    """更新Scaffold相关的参数名称以兼容GradSampleModule"""
    # GradSampleModule会添加_module前缀，需要同步更新
    self.regular_params_name = list(
        key for key, _ in self.model.named_parameters()
    )
    
    # 确保控制变量的维度匹配
    if hasattr(self, 'c_local') and self.c_local is not None:
        self._align_control_variates_with_wrapped_model()
```

### 阶段5: DPScaffoldStein最终实现

#### 5.1 完整组合设计
```python
# src/client/dp_scaffold_stein.py
from src.client.dp_scaffold import DPScaffoldClient

class DPScaffoldSteinClient(DPScaffoldClient):
    """完整的DP + Scaffold + JSE实现"""
    
    def __init__(self, **commons):
        super().__init__(**commons)
        # 默认启用JSE
        self.jse_enabled = True
        
        # 算法变体配置，默认为last_noise_server_jse（对应论文主要算法）
        variant_config = getattr(
            self.args.dp_scaffold_stein, 'algorithm_variant', 'last_noise_server_jse'
        )
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
    
    def fit(self):
        """实现三种完整的DP-ScaffStein算法变体"""
        if self.algorithm_variant == 1:  # last_noise_server_jse
            return self._dp_scaffstein_last_noise_server_jse()
        elif self.algorithm_variant == 2:  # step_noise_step_jse
            return self._dp_scaffstein_step_noise_step_jse()
        elif self.algorithm_variant == 3:  # step_noise_final_jse
            return self._dp_scaffstein_step_noise_final_jse()
    
    def _dp_scaffstein_last_noise_server_jse(self):
        """last_noise_server_jse: 最后加噪声 + 服务器JSE"""
        # 标准Scaffold训练（无噪声）
        for k in range(self.local_epoch):
            x, y = self.get_data_batch()
            self._scaffold_training_step_no_noise(x, y)
        
        # 最后添加参数级噪声
        param_diff = self._get_parameter_difference()
        noisy_param_diff = self._add_parameter_level_noise_with_scaling(
            param_diff, k_factor=self.local_epoch
        )
        
        # 更新控制变量
        self._update_control_variates_dp(noisy_param_diff)
        
        return noisy_param_diff  # 服务器端应用JSE
    
    def _dp_scaffstein_step_noise_step_jse(self):
        """step_noise_step_jse: 每步加噪声 + 每步JSE"""
        for k in range(self.local_epoch):
            x, y = self.get_data_batch()
            
            # DP处理
            self._compute_per_sample_gradients(x, y)
            self._clip_and_add_noise_opacus()
            
            # 每步JSE
            self._apply_jse_to_current_gradients()
            
            # Scaffold修正
            self._apply_scaffold_correction()
            
            # 参数更新
            self.optimizer.step()
        
        # 标准控制变量更新
        param_diff = self._get_parameter_difference()
        self._update_control_variates_standard(param_diff)
        
        return param_diff
    
    def _dp_scaffstein_step_noise_final_jse(self):
        """step_noise_final_jse: 每步加噪声 + 最后JSE"""
        for k in range(self.local_epoch):
            x, y = self.get_data_batch()
            
            # DP处理
            self._compute_per_sample_gradients(x, y)
            self._clip_and_add_noise_opacus()
            
            # Scaffold修正
            self._apply_scaffold_correction()
            
            # 参数更新
            self.optimizer.step()
        
        # 最后对参数差值应用JSE
        param_diff = self._get_parameter_difference()
        jse_param_diff = self._apply_jse_to_parameter_diff(
            param_diff, k_factor=self.local_epoch
        )
        
        # 控制变量更新使用原始参数差值
        self._update_control_variates_standard(param_diff)
        
        return jse_param_diff
```

#### 5.2 服务器端对应实现
```python
# src/server/dp_scaffold_stein.py
from src.server.dp_scaffold import DPScaffoldServer
from src.utils.jse_utils import JSEProcessor

class DPScaffoldSteinServer(DPScaffoldServer):
    client_cls = DPScaffoldSteinClient
    
    def aggregate_client_updates(self, client_packages):
        # 变体1需要服务器端JSE处理
        if self._get_algorithm_variant() == 1:
            self._apply_server_side_jse_variant_1(client_packages)
        
        # 调用父类的Scaffold聚合逻辑
        super().aggregate_client_updates(client_packages)
    
    def _apply_server_side_jse_variant_1(self, client_packages):
        """为算法变体1应用服务器端JSE"""
        # 获取聚合前的参数更新
        aggregated_update = self._compute_aggregated_update(client_packages)
        
        # 计算JSE所需的噪声方差
        noise_variance = self._compute_server_noise_variance()
        
        # 应用JSE
        jse_update = JSEProcessor.apply_jse_shrinkage(
            aggregated_update, noise_variance
        )
        
        # 更新客户端包中的参数更新
        self._update_client_packages_with_jse(client_packages, jse_update)
```

## 5. 配置系统设计

### 5.1 配置文件结构

```yaml
# config/method/dp_fedavg_local.yaml (扩展现有)
defaults:
  - /client: dp_fedavg_local
  - /server: dp_fedavg_local

dp_fedavg_local:
  algorithm_variant: "step_noise"  # 或数字 2 (可选: last_noise, step_noise)
  clip_norm: 1.0            # 裁剪阈值
  sigma: 1.0                # 噪声标准差参数

# config/method/dp_fed_stein.yaml (新增)
defaults:
  - /client: dp_fed_stein
  - /server: dp_fed_stein

dp_fed_stein:
  algorithm_variant: "step_noise_step_jse"  # 默认变体
  jse_enabled: true          # 默认启用JSE
  clip_norm: 1.0
  sigma: 1.0

# config/method/dp_scaffold.yaml (新增)
defaults:
  - /client: dp_scaffold
  - /server: dp_scaffold

dp_scaffold:
  algorithm_variant: "step_noise_step_jse"
  jse_enabled: false
  clip_norm: 1.0
  sigma: 1.0
  global_lr: 1.0            # Scaffold全局学习率

# config/method/dp_scaffold_stein.yaml (新增)
defaults:
  - /client: dp_scaffold_stein
  - /server: dp_scaffold_stein

dp_scaffold_stein:
  algorithm_variant: "last_noise_server_jse"  # 默认变体（对应论文主要算法）
  jse_enabled: true          # 默认启用JSE
  clip_norm: 1.0
  sigma: 1.0
  global_lr: 1.0
```

### 5.2 配置验证和自动推导

```python
# src/utils/config_validator.py
class DPMethodConfigValidator:
    """DP方法配置验证器"""
    
    @staticmethod
    def validate_dp_config(config):
        """验证DP配置的合理性"""
        assert config.clip_norm > 0, "clip_norm must be positive"
        assert config.sigma > 0, "sigma must be positive"
        assert config.algorithm_variant in [1, 2, 3], "algorithm_variant must be 1, 2, or 3"
    
    @staticmethod
    def auto_derive_noise_parameters(config):
        """自动推导噪声相关参数"""
        # 根据算法变体自动设置噪声缩放因子
        variant = config.algorithm_variant
        if isinstance(variant, str):
            if variant == "last_noise_server_jse":
                config.noise_scaling_factor = "K_eta_l"  # K * η_l
            else:
                config.noise_scaling_factor = "standard"  # 1
        else:
            if variant == 1:
                config.noise_scaling_factor = "K_eta_l"  # K * η_l
            else:
                config.noise_scaling_factor = "standard"  # 1
        
        return config
```

## 6. 关键技术细节

### 6.1 噪声标准差计算

不同算法变体的噪声标准差计算：

```python
def compute_noise_std(self, algorithm_variant, batch_size: int) -> float:
    """计算噪声标准差"""
    # 支持字符串和数字配置
    if isinstance(algorithm_variant, str):
        is_last_noise = algorithm_variant == "last_noise_server_jse"
    else:
        is_last_noise = algorithm_variant == 1
        
    if is_last_noise:
        # 参数级噪声：σ_DP = C * K * η_l * σ_g / b
        K = self.local_epoch
        eta_l = self.args.optimizer.lr
        return self.clip_norm * K * eta_l * self.sigma / batch_size
    else:
        # 梯度级噪声：σ_DP = C * σ_g / b
        return self.clip_norm * self.sigma / batch_size
```

### 6.2 JSE数值稳定性处理

```python
def apply_jse_with_stability(tensor: torch.Tensor, noise_variance: float) -> torch.Tensor:
    """带数值稳定性检查的JSE应用"""
    d = tensor.numel()
    tensor_norm_sq = tensor.norm().item() ** 2
    
    # 数值稳定性检查
    if tensor_norm_sq < 1e-8:
        return tensor
    
    # 计算收缩因子
    shrinkage_factor = (d - 2) * noise_variance / tensor_norm_sq
    shrinkage_factor = torch.clamp(shrinkage_factor, 0.0, 1.0)
    
    return (1.0 - shrinkage_factor) * tensor
```

### 6.3 控制变量与DP的兼容性

```python
def update_control_variates_with_dp(self, param_diff: torch.Tensor, is_noisy: bool = False):
    """更新控制变量，考虑DP噪声的影响"""
    if is_noisy:
        # 如果参数差值包含噪声，需要特殊处理
        # 控制变量更新应基于真实的参数变化，而非噪声版本
        clean_param_diff = self._get_clean_parameter_difference()
        self._update_control_variates_standard(clean_param_diff)
    else:
        # 标准控制变量更新
        self._update_control_variates_standard(param_diff)
```

## 7. 测试和验证策略

### 7.1 单元测试设计

```python
# tests/test_dp_methods.py
class TestDPMethods:
    def test_noise_variance_calculation(self):
        """测试噪声方差计算的正确性"""
        pass
    
    def test_jse_shrinkage_correctness(self):
        """测试JSE收缩的数学正确性"""
        pass
    
    def test_algorithm_variant_consistency(self):
        """测试算法变体的一致性"""
        pass
    
    def test_scaffold_dp_integration(self):
        """测试Scaffold与DP的集成"""
        pass
```

### 7.2 集成测试

```python
def test_end_to_end_training():
    """端到端训练测试"""
    # 测试四种方法的三种变体
    methods = ['dp_fedavg_local', 'dp_fed_stein', 'dp_scaffold', 'dp_scaffold_stein']
    variants = [1, 2, 3]
    
    for method in methods:
        for variant in variants:
            config = create_test_config(method, variant)
            result = run_federated_training(config)
            assert_training_success(result)
```

## 8. 实施时间表

### 第1周：DPFedAvg扩展
- 扩展现有`DPFedAvgLocalClient`支持三种变体
- 实现基础的变体切换逻辑
- 添加参数级噪声处理（变体1）

### 第2周：JSE工具模块
- 实现`JSEProcessor`工具类
- 集成JSE到DPFedAvg各变体
- 创建`DPFedSteinClient`

### 第3周：DPScaffold实现
- 基于`SCAFFOLDClient`实现`DPScaffoldClient`
- 处理控制变量与DP的兼容性
- 实现三种算法变体

### 第4周：DPScaffoldStein最终实现
- 实现`DPScaffoldSteinClient`
- 完成所有服务器端对应实现
- 全面测试和调试

### 第5周：测试和文档
- 单元测试和集成测试
- 性能基准测试
- 文档完善和示例代码

## 9. 文件组织结构

```
src/
├── client/
│   ├── fedavg.py                    # 基础 (已存在)
│   ├── scaffold.py                  # Scaffold (已存在)
│   ├── dp_fedavg_local.py          # DP-FedAvg (已存在，需扩展)
│   ├── dp_fed_stein.py             # 新增
│   ├── dp_scaffold.py              # 新增
│   └── dp_scaffold_stein.py        # 新增
├── server/
│   ├── fedavg.py                    # 基础 (已存在)
│   ├── scaffold.py                  # Scaffold (已存在)
│   ├── dp_fedavg_local.py          # DP-FedAvg服务器 (已存在，需扩展)
│   ├── dp_fed_stein.py             # 新增
│   ├── dp_scaffold.py              # 新增
│   └── dp_scaffold_stein.py        # 新增
├── utils/
│   ├── jse_utils.py                 # JSE工具模块 (新增)
│   └── config_validator.py         # 配置验证 (新增)
config/
├── method/
│   ├── dp_fedavg_local.yaml        # 扩展现有
│   ├── dp_fed_stein.yaml           # 新增
│   ├── dp_scaffold.yaml            # 新增
│   └── dp_scaffold_stein.yaml      # 新增
```

## 10. 总结

本设计文档提供了FL-bench框架中四种差分隐私联邦学习方法的完整实现方案。通过分阶段的渐进式开发，我们能够：

1. **最大化代码复用**：基于现有的`DPFedAvgLocalClient`和`SCAFFOLDClient`
2. **保持架构一致性**：遵循FL-bench的设计模式和继承结构
3. **提供灵活配置**：支持三种算法变体和可选JSE启用
4. **确保数学正确性**：严格按照论文算法实现噪声添加和JSE处理
5. **便于维护扩展**：模块化设计便于后续新算法的添加

每个阶段都有明确的目标和可验证的输出，确保实施过程的可控性和质量保证。

---

*文档版本: 1.0*  
*创建日期: 2025-09-12*  
*最后更新: 2025-09-12*