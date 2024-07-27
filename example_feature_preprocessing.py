#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征预处理使用示例

这个示例展示了如何在训练模型前使用基于相关性和缺失值的特征预处理算法
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from alpha.simple_factor_model import SimpleFactorModel

def main():
    """主函数：演示特征预处理的使用"""
    
    print("=" * 70)
    print("特征预处理使用示例")
    print("=" * 70)
    
    # 1. 创建模型实例
    print("\n1. 创建模型实例...")
    model = SimpleFactorModel(
        train_start="20150101",
        train_end="20221231",
        test_start="20230101", 
        test_end="20241231",
        n_groups=2,
        future_days=1,
        data_frequency="W"  # 使用周频数据
    )
    
    # 2. 加载和准备数据
    print("\n2. 加载和准备数据...")
    model.load_and_prepare_data()
    
    print(f"原始特征数量: {model.train_X.shape[1] if model.train_X is not None else 0}")
    
    # 3. 方法一：在训练前手动进行特征预处理
    print("\n3. 方法一：手动特征预处理...")
    print("-" * 40)
    
    # 执行特征预处理
    kept_features, removed_features = model.preprocess_features(
        threshold=0.6,  # 相关性阈值
        plot_correlation_matrix=True  # 显示相关性矩阵图
    )
    
    # 打印预处理结果
    model.print_feature_preprocessing_summary()
    
    # 4. 方法二：在训练时自动进行特征预处理
    print("\n4. 方法二：训练时自动特征预处理...")
    print("-" * 40)
    
    # 创建新的模型实例来演示自动预处理
    model_auto = SimpleFactorModel(
        train_start="20150101",
        train_end="20221231",
        test_start="20230101",
        test_end="20241231",
        n_groups=2,
        future_days=1,
        data_frequency="W"
    )
    
    # 加载数据
    model_auto.load_and_prepare_data()
    
    # 训练模型时自动进行特征预处理
    model_auto.train_model(
        preprocess_features=True,  # 启用自动特征预处理
        correlation_threshold=0.7  # 设置相关性阈值
    )
    
    # 打印预处理信息
    model_auto.print_feature_preprocessing_summary()
    
    # 5. 评估模型性能
    print("\n5. 评估模型性能...")
    print("-" * 40)
    model_auto.evaluate_model()
    
    # 6. 保存模型（包含预处理信息）
    print("\n6. 保存模型...")
    print("-" * 40)
    save_info = model_auto.save_model(model_name="example_preprocessing_model")
    print(f"模型已保存到: {save_info['model_file']}")
    
    # 7. 加载模型并验证预处理信息
    print("\n7. 加载模型并验证预处理信息...")
    print("-" * 40)
    
    loaded_model = SimpleFactorModel()
    model_info = loaded_model.load_model(model_name="example_preprocessing_model")
    
    # 验证预处理信息
    preprocessing_info = loaded_model.get_feature_preprocessing_info()
    if preprocessing_info:
        print("✓ 特征预处理信息成功加载")
        print(f"  原始特征数量: {len(preprocessing_info['original_features'])}")
        print(f"  保留特征数量: {len(preprocessing_info['kept_features'])}")
        print(f"  移除特征数量: {len(preprocessing_info['removed_features'])}")
        print(f"  相关性阈值: {preprocessing_info['correlation_threshold']}")
    else:
        print("✗ 特征预处理信息加载失败")
    
    print("\n" + "=" * 70)
    print("特征预处理使用示例完成")
    print("=" * 70)

def demonstrate_algorithm():
    """演示特征预处理算法的核心逻辑"""
    
    print("\n" + "=" * 70)
    print("特征预处理算法核心逻辑演示")
    print("=" * 70)
    
    print("""
算法步骤：
1. 计算每个特征的缺失值数量
2. 计算特征间的相关系数矩阵
3. 使用图算法找到高度相关的特征组（连通分量）
4. 在每个连通分量中，保留缺失值最少的特征
5. 移除其他高度相关的特征

优势：
- 减少特征间的多重共线性
- 保留数据质量最好的特征
- 降低模型复杂度
- 提高训练效率
- 减少过拟合风险
    """)

if __name__ == "__main__":
    # 演示算法逻辑
    demonstrate_algorithm()
    
    # 运行完整示例
    main() 