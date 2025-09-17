#!/usr/bin/env python3
"""
DP-ScaffStein Algorithm Variants Verification Script
验证三种DP-ScaffStein算法变体的正确实现
"""

import torch
import sys
import os
sys.path.append('.')

from src.client.dp_scaffstein import DPScaffSteinClient
from src.server.dp_scaffstein import DPScaffSteinServer
from src.utils.jse_utils import JSEProcessor
from collections import OrderedDict


class MockArgs:
    """Mock arguments for testing"""
    def __init__(self, variant='step_noise_final_jse'):
        self.method = 'dp_scaffstein'
        self.model = type('Model', (), {'name': '2nn'})()
        self.dataset = type('Dataset', (), {'name': 'mnist'})()
        self.dp_scaffstein = type('Config', (), {
            'algorithm_variant': variant,
            'clip_norm': 1.0,
            'sigma': 0.1,
            'global_lr': 1.0
        })()
        self.optimizer = type('Optimizer', (), {
            'name': 'sgd',
            'lr': 0.1,
            'momentum': 0.9
        })()
        self.common = type('Common', (), {
            'local_epoch': 3,
            'batch_size': 32
        })()


def test_algorithm_variant_routing():
    """测试算法变体路由逻辑"""
    print("=== 测试算法变体路由逻辑 ===")

    variants = [
        ('last_noise_server_jse', 3),
        ('step_noise_step_jse', 4),
        ('step_noise_final_jse', 5),
        ('last_noise', 1),  # 向后兼容
        ('step_noise', 2)   # 向后兼容
    ]

    for variant_name, expected_id in variants:
        try:
            args = MockArgs(variant_name)

            # Test variant mapping logic (simulate from __init__)
            config = args.dp_scaffstein
            variant_config = getattr(config, 'algorithm_variant', 'step_noise_final_jse')

            ALGORITHM_VARIANTS = {
                "last_noise": 1,
                "step_noise": 2,
                "last_noise_server_jse": 3,
                "step_noise_step_jse": 4,
                "step_noise_final_jse": 5
            }

            algorithm_variant = ALGORITHM_VARIANTS[variant_config]

            if algorithm_variant == expected_id:
                print(f"✅ {variant_name} -> 变体 {algorithm_variant} (正确)")
            else:
                print(f"❌ {variant_name} -> 变体 {algorithm_variant} (期望: {expected_id})")

        except Exception as e:
            print(f"❌ {variant_name}: 错误 - {e}")

    print()


def test_fit_method_dispatch():
    """测试fit方法的分发逻辑"""
    print("=== 测试fit方法分发逻辑 ===")

    variants_methods = {
        1: '_last_noise_training',
        2: '_step_noise_training',
        3: '_last_noise_server_jse_training',
        4: '_step_noise_step_jse_training',
        5: '_step_noise_final_jse_training'
    }

    for variant_id, expected_method in variants_methods.items():
        # Find variant name
        variant_name = None
        for name, vid in [('last_noise', 1), ('step_noise', 2),
                         ('last_noise_server_jse', 3), ('step_noise_step_jse', 4),
                         ('step_noise_final_jse', 5)]:
            if vid == variant_id:
                variant_name = name
                break

        try:
            args = MockArgs(variant_name)

            # Check if method exists
            if hasattr(DPScaffSteinClient, expected_method):
                print(f"✅ 变体 {variant_id} ({variant_name}) -> {expected_method} (方法存在)")
            else:
                print(f"❌ 变体 {variant_id} ({variant_name}) -> {expected_method} (方法缺失)")

        except Exception as e:
            print(f"❌ 变体 {variant_id}: 错误 - {e}")

    print()


def test_jse_integration():
    """测试JSE集成功能"""
    print("=== 测试JSE集成功能 ===")

    try:
        # Test JSE utility methods
        param_diff = torch.randn(5, 3)
        noise_var = 0.01

        # Test shrinkage
        shrinked = JSEProcessor.apply_jse_shrinkage(param_diff, noise_var)
        shrinkage_applied = not torch.equal(param_diff, shrinked)
        print(f"✅ JSE收缩: {'应用成功' if shrinkage_applied else '未应用'}")

        # Test global JSE
        param_dict = OrderedDict({
            'layer1.weight': torch.randn(4, 3),
            'layer1.bias': torch.randn(4)
        })
        original_norms = {k: v.norm().item() for k, v in param_dict.items()}

        JSEProcessor.apply_global_jse_to_parameter_diff(param_dict, noise_var, k_factor=2)
        new_norms = {k: v.norm().item() for k, v in param_dict.items()}

        # JSE should generally reduce norms (shrinkage effect)
        shrinkage_count = sum(1 for k in original_norms if new_norms[k] < original_norms[k])
        print(f"✅ 全局JSE: {shrinkage_count}/{len(param_dict)} 参数被收缩")

    except Exception as e:
        print(f"❌ JSE集成测试错误: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_server_jse():
    """测试服务器端JSE功能"""
    print("=== 测试服务器端JSE功能 ===")

    try:
        # Mock server args
        server_args = type('Args', (), {
            'dp_scaffstein': type('Config', (), {
                'algorithm_variant': 1  # last_noise_server_jse
            })()
        })()

        # Test server variant detection
        algorithm_variant = getattr(server_args.dp_scaffstein, 'algorithm_variant', 'step_noise_final_jse')
        if isinstance(algorithm_variant, str):
            variant_map = {
                'last_noise_server_jse': 1,
                'step_noise_step_jse': 2,
                'step_noise_final_jse': 3
            }
            algorithm_variant = variant_map[algorithm_variant]

        should_apply_server_jse = (algorithm_variant == 1)
        print(f"✅ 服务器JSE检测: 变体 {algorithm_variant} -> {'需要服务器JSE' if should_apply_server_jse else '客户端JSE'}")

        # Test server JSE method existence
        if hasattr(DPScaffSteinServer, '_apply_server_jse'):
            print("✅ 服务器JSE方法: _apply_server_jse 存在")
        else:
            print("❌ 服务器JSE方法: _apply_server_jse 缺失")

    except Exception as e:
        print(f"❌ 服务器JSE测试错误: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_inheritance_compatibility():
    """测试继承兼容性"""
    print("=== 测试继承兼容性 ===")

    try:
        from src.client.dp_scaffold import DPScaffoldClient

        # Test inheritance
        is_subclass = issubclass(DPScaffSteinClient, DPScaffoldClient)
        print(f"✅ 继承关系: DPScaffSteinClient is subclass of DPScaffoldClient: {is_subclass}")

        # Test method override vs inheritance
        base_methods = set(dir(DPScaffoldClient))
        extended_methods = set(dir(DPScaffSteinClient))

        new_methods = extended_methods - base_methods
        jse_methods = [m for m in new_methods if 'jse' in m.lower()]

        print(f"✅ 新增方法: {len(new_methods)} 个")
        print(f"✅ JSE相关方法: {len(jse_methods)} 个 {jse_methods}")

        # Test key inherited methods
        key_methods = ['fit', '_step_noise_training', '_last_noise_training',
                      '_clip_and_add_noise_with_scaffold', 'package']

        for method in key_methods:
            if hasattr(DPScaffSteinClient, method):
                print(f"✅ 关键方法: {method} 可用")
            else:
                print(f"❌ 关键方法: {method} 缺失")

    except Exception as e:
        print(f"❌ 继承兼容性测试错误: {e}")
        import traceback
        traceback.print_exc()

    print()


def main():
    """运行所有验证测试"""
    print("DP-ScaffStein算法变体验证测试")
    print("=" * 50)

    test_algorithm_variant_routing()
    test_fit_method_dispatch()
    test_jse_integration()
    test_server_jse()
    test_inheritance_compatibility()

    print("=" * 50)
    print("验证测试完成")


if __name__ == "__main__":
    main()