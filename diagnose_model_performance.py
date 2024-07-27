#!/usr/bin/env python3
"""
二分类模型性能诊断脚本
用于分析样本内回测效果不好的原因
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_label_quality(train_y, test_y):
    """诊断标签质量"""
    print("=" * 60)
    print("标签质量诊断")
    print("=" * 60)
    
    # 标签分布
    train_dist = train_y.value_counts().sort_index()
    test_dist = test_y.value_counts().sort_index()
    
    print(f"训练集标签分布: {dict(train_dist)}")
    print(f"测试集标签分布: {dict(test_dist)}")
    
    # 计算不平衡比例
    train_imbalance = train_dist.max() / train_dist.min()
    test_imbalance = test_dist.max() / test_dist.min()
    
    print(f"训练集不平衡比例: {train_imbalance:.2f}")
    print(f"测试集不平衡比例: {test_imbalance:.2f}")
    
    # 标签稳定性分析
    if hasattr(train_y, 'index') and hasattr(train_y.index, 'get_level_values'):
        dates = train_y.index.get_level_values('date').unique()
        print(f"训练集时间跨度: {len(dates)} 个交易日")
        
        # 计算每个日期的标签分布
        daily_distributions = []
        for date in dates[:10]:  # 只分析前10天作为示例
            daily_labels = train_y.xs(date, level='date')
            daily_dist = daily_labels.value_counts().sort_index()
            daily_distributions.append(daily_dist)
        
        if daily_distributions:
            print("\n前10天每日标签分布:")
            for i, dist in enumerate(daily_distributions):
                print(f"  第{i+1}天: {dict(dist)}")
    
    return {
        'train_imbalance': train_imbalance,
        'test_imbalance': test_imbalance,
        'train_dist': train_dist,
        'test_dist': test_dist
    }

def diagnose_feature_quality(train_X, test_X):
    """诊断特征质量"""
    print("\n" + "=" * 60)
    print("特征质量诊断")
    print("=" * 60)
    
    print(f"特征数量: {train_X.shape[1]}")
    print(f"训练集样本数: {train_X.shape[0]}")
    print(f"测试集样本数: {test_X.shape[0]}")
    
    # 缺失值分析
    train_missing = train_X.isnull().sum()
    test_missing = test_X.isnull().sum()
    
    print(f"\n缺失值统计:")
    print(f"  训练集缺失值总数: {train_missing.sum()}")
    print(f"  测试集缺失值总数: {test_missing.sum()}")
    print(f"  训练集缺失比例: {train_missing.sum() / (train_X.shape[0] * train_X.shape[1]) * 100:.2f}%")
    print(f"  测试集缺失比例: {test_missing.sum() / (test_X.shape[0] * test_X.shape[1]) * 100:.2f}%")
    
    # 异常值分析
    train_stats = train_X.describe()
    print(f"\n特征统计信息:")
    print(f"  特征均值范围: [{train_stats.loc['mean'].min():.4f}, {train_stats.loc['mean'].max():.4f}]")
    print(f"  特征标准差范围: [{train_stats.loc['std'].min():.4f}, {train_stats.loc['std'].max():.4f}]")
    
    # 相关性分析
    print(f"\n特征相关性分析:")
    corr_matrix = train_X.corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    print(f"  高相关性特征对数量 (|corr| > 0.8): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print("  前5个高相关性特征对:")
        for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:5]):
            print(f"    {feat1} - {feat2}: {corr:.3f}")
    
    return {
        'feature_count': train_X.shape[1],
        'missing_ratio': train_missing.sum() / (train_X.shape[0] * train_X.shape[1]),
        'high_corr_pairs': len(high_corr_pairs)
    }

def diagnose_model_performance(model, train_X, train_y, test_X, test_y):
    """诊断模型性能"""
    print("\n" + "=" * 60)
    print("模型性能诊断")
    print("=" * 60)
    
    # 训练集性能
    train_pred = model.predict(train_X)
    train_pred_proba = model.predict_proba(train_X)[:, 1]
    
    train_accuracy = accuracy_score(train_y, train_pred)
    train_roc_auc = roc_auc_score(train_y, train_pred_proba)
    train_f1 = f1_score(train_y, train_pred)
    
    # 测试集性能
    test_pred = model.predict(test_X)
    test_pred_proba = model.predict_proba(test_X)[:, 1]
    
    test_accuracy = accuracy_score(test_y, test_pred)
    test_roc_auc = roc_auc_score(test_y, test_pred_proba)
    test_f1 = f1_score(test_y, test_pred)
    
    print(f"训练集性能:")
    print(f"  准确率: {train_accuracy:.4f}")
    print(f"  ROC AUC: {train_roc_auc:.4f}")
    print(f"  F1 Score: {train_f1:.4f}")
    
    print(f"\n测试集性能:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  ROC AUC: {test_roc_auc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # 过拟合检测
    accuracy_gap = train_accuracy - test_accuracy
    roc_auc_gap = train_roc_auc - test_roc_auc
    f1_gap = train_f1 - test_f1
    
    print(f"\n过拟合检测:")
    print(f"  准确率差距: {accuracy_gap:.4f}")
    print(f"  ROC AUC差距: {roc_auc_gap:.4f}")
    print(f"  F1 Score差距: {f1_gap:.4f}")
    
    if accuracy_gap > 0.1 or roc_auc_gap > 0.1:
        print("  ⚠️  可能存在过拟合问题")
    else:
        print("  ✅ 过拟合风险较低")
    
    # 预测概率分布
    print(f"\n预测概率分布:")
    print(f"  训练集平均概率: {train_pred_proba.mean():.4f}")
    print(f"  测试集平均概率: {test_pred_proba.mean():.4f}")
    print(f"  训练集概率标准差: {train_pred_proba.std():.4f}")
    print(f"  测试集概率标准差: {test_pred_proba.std():.4f}")
    
    return {
        'train_metrics': {'accuracy': train_accuracy, 'roc_auc': train_roc_auc, 'f1': train_f1},
        'test_metrics': {'accuracy': test_accuracy, 'roc_auc': test_roc_auc, 'f1': test_f1},
        'overfitting_gaps': {'accuracy': accuracy_gap, 'roc_auc': roc_auc_gap, 'f1': f1_gap}
    }

def analyze_feature_importance(model, train_X, train_y):
    """分析特征重要性"""
    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)
    
    if hasattr(model, 'feature_importances_'):
        # 获取特征重要性
        importance = pd.DataFrame({
            'feature': train_X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"前10个最重要特征:")
        for i, (_, row) in enumerate(importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        print(f"\n后10个最不重要特征:")
        for i, (_, row) in enumerate(importance.tail(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # 计算重要性分布
        importance_stats = importance['importance'].describe()
        print(f"\n特征重要性统计:")
        print(f"  均值: {importance_stats['mean']:.4f}")
        print(f"  标准差: {importance_stats['std']:.4f}")
        print(f"  最大值: {importance_stats['max']:.4f}")
        print(f"  最小值: {importance_stats['min']:.4f}")
        
        # 零重要性特征
        zero_importance = (importance['importance'] == 0).sum()
        print(f"  零重要性特征数量: {zero_importance}")
        
        return importance
    else:
        print("模型不支持特征重要性分析")
        return None

def suggest_improvements(label_diagnosis, feature_diagnosis, model_diagnosis, feature_importance):
    """基于诊断结果提出改进建议"""
    print("\n" + "=" * 60)
    print("改进建议")
    print("=" * 60)
    
    suggestions = []
    
    # 标签质量改进建议
    if label_diagnosis['train_imbalance'] > 1.5:
        suggestions.append("标签不平衡问题: 考虑使用SMOTE或调整类别权重")
    
    # 特征质量改进建议
    if feature_diagnosis['missing_ratio'] > 0.1:
        suggestions.append("缺失值过多: 改进缺失值处理策略")
    
    if feature_diagnosis['high_corr_pairs'] > 100:
        suggestions.append("特征相关性过高: 考虑特征选择或降维")
    
    if feature_diagnosis['feature_count'] > 50:
        suggestions.append("特征数量过多: 考虑特征选择减少维度")
    
    # 模型性能改进建议
    if model_diagnosis['overfitting_gaps']['accuracy'] > 0.1:
        suggestions.append("过拟合严重: 增加正则化或减少模型复杂度")
    
    if model_diagnosis['test_metrics']['roc_auc'] < 0.55:
        suggestions.append("模型性能较差: 检查标签质量和特征工程")
    
    # 特征重要性改进建议
    if feature_importance is not None:
        zero_importance = (feature_importance['importance'] == 0).sum()
        if zero_importance > feature_diagnosis['feature_count'] * 0.3:
            suggestions.append("大量无用特征: 进行特征选择")
    
    if not suggestions:
        print("✅ 当前模型配置基本合理，建议进行样本外测试验证")
    else:
        print("发现以下问题，建议优先处理:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    return suggestions

def main():
    """主函数"""
    print("二分类模型性能诊断工具")
    print("=" * 60)
    
    try:
        # 导入模型
        from alpha.simple_factor_model import SimpleFactorModel
        
        # 创建模型实例
        model = SimpleFactorModel(
            train_start="20220101",
            train_end="20221231",
            test_start="20230101",
            test_end="20231231",
            n_groups=2,
            future_days=5,
            data_frequency="D"
        )
        
        # 加载数据
        print("正在加载数据...")
        train_X, train_y, test_X, test_y = model.load_and_prepare_data()
        
        if train_X is None or train_y is None:
            print("❌ 数据加载失败")
            return
        
        # 训练模型
        print("正在训练模型...")
        trained_model = model.train_model()
        
        # 执行诊断
        label_diagnosis = diagnose_label_quality(train_y, test_y)
        feature_diagnosis = diagnose_feature_quality(train_X, test_X)
        model_diagnosis = diagnose_model_performance(trained_model, train_X, train_y, test_X, test_y)
        feature_importance = analyze_feature_importance(trained_model, train_X, train_y)
        
        # 提出改进建议
        suggestions = suggest_improvements(label_diagnosis, feature_diagnosis, model_diagnosis, feature_importance)
        
        print("\n" + "=" * 60)
        print("诊断完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 诊断过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 