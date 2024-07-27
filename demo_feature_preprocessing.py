#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征预处理功能演示

这个脚本演示了如何使用基于相关性和缺失值的特征预处理算法
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def demo_basic_usage():
    """演示基本使用方法"""
    
    print("=" * 60)
    print("特征预处理功能演示")
    print("=" * 60)
    
    try:
        from alpha.simple_factor_model import SimpleFactorModel
        
        # 创建模型实例
        print("\n1. 创建模型实例...")
        model = SimpleFactorModel(
            train_start="20200101",
            train_end="20221231",
            test_start="20230101",
            test_end="20231231",
            n_groups=2,
            future_days=5,
            data_frequency="W"  # 使用周频数据，减少计算时间
        )
        
        # 加载数据
        print("\n2. 加载数据...")
        model.load_and_prepare_data()
        
        print(f"原始特征数量: {model.train_X.shape[1] if model.train_X is not None else 0}")
        
        # 执行特征预处理
        print("\n3. 执行特征预处理...")
        kept_features, removed_features = model.preprocess_features(
            threshold=0.6,
            plot_correlation_matrix=False  # 设置为False避免显示图形
        )
        
        print(f"保留特征数量: {len(kept_features)}")
        print(f"移除特征数量: {len(removed_features)}")
        
        # 打印预处理摘要
        print("\n4. 预处理摘要:")
        model.print_feature_preprocessing_summary()
        
        # 训练模型
        print("\n5. 训练模型...")
        model.train_model(preprocess_features=False)  # 已经预处理过了
        
        # 评估模型
        print("\n6. 评估模型...")
        model.evaluate_model()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def demo_auto_preprocessing():
    """演示自动预处理功能"""
    
    print("\n" + "=" * 60)
    print("自动特征预处理演示")
    print("=" * 60)
    
    try:
        from alpha.simple_factor_model import SimpleFactorModel
        
        # 创建模型实例
        print("\n1. 创建模型实例...")
        model = SimpleFactorModel(
            train_start="20200101",
            train_end="20221231",
            test_start="20230101",
            test_end="20231231",
            n_groups=2,
            future_days=5,
            data_frequency="W"
        )
        
        # 加载数据
        print("\n2. 加载数据...")
        model.load_and_prepare_data()
        
        print(f"原始特征数量: {model.train_X.shape[1] if model.train_X is not None else 0}")
        
        # 训练模型时自动进行特征预处理
        print("\n3. 训练模型（自动特征预处理）...")
        model.train_model(
            preprocess_features=True,  # 启用自动特征预处理
            correlation_threshold=0.7  # 设置相关性阈值
        )
        
        # 打印预处理摘要
        print("\n4. 预处理摘要:")
        model.print_feature_preprocessing_summary()
        
        # 评估模型
        print("\n5. 评估模型...")
        model.evaluate_model()
        
        print("\n" + "=" * 60)
        print("自动预处理演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 演示基本使用方法
    demo_basic_usage()
    
    # 演示自动预处理功能
    demo_auto_preprocessing() 