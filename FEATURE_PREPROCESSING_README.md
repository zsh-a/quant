# 特征预处理功能说明

## 概述

本功能实现了基于相关性和缺失值的特征预处理算法，用于在训练模型前自动选择最优特征子集，减少特征间的多重共线性，提高模型性能。

## 算法原理

### 核心思想
1. **计算缺失值**：统计每个特征的缺失值数量
2. **计算相关性**：计算特征间的相关系数矩阵
3. **图算法聚类**：使用深度优先搜索(DFS)找到高度相关的特征组（连通分量）
4. **特征选择**：在每个连通分量中，保留缺失值最少的特征，移除其他特征

### 算法步骤
```python
# 1. 计算每个特征的缺失值数量
missing_counts = df[features].isnull().sum().to_dict()

# 2. 计算特征间的相关系数矩阵
corr_matrix = df[features].corr()

# 3. 构建相关性图
graph = defaultdict(list)
for i in range(n):
    for j in range(i + 1, n):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            graph[feature_i].append(feature_j)
            graph[feature_j].append(feature_i)

# 4. 使用DFS找到连通分量
components = []
visited = set()
for feature in features:
    if feature not in visited:
        comp = []
        dfs(feature, comp)
        components.append(comp)

# 5. 在每个连通分量中保留缺失值最少的特征
for comp in components:
    if len(comp) == 1:
        to_keep.append(comp[0])
    else:
        best_feature = min(comp, key=lambda x: missing_counts[x])
        to_keep.append(best_feature)
        to_remove.extend([f for f in comp if f != best_feature])
```

## 使用方法

### 方法一：手动特征预处理

```python
from alpha.simple_factor_model import SimpleFactorModel

# 创建模型实例
model = SimpleFactorModel(
    train_start="20150101",
    train_end="20221231",
    test_start="20230101",
    test_end="20241231"
)

# 加载数据
model.load_and_prepare_data()

# 执行特征预处理
kept_features, removed_features = model.preprocess_features(
    threshold=0.6,  # 相关性阈值
    plot_correlation_matrix=True  # 是否显示相关性矩阵图
)

# 打印预处理摘要
model.print_feature_preprocessing_summary()

# 训练模型
model.train_model(preprocess_features=False)  # 已经预处理过了
```

### 方法二：训练时自动特征预处理

```python
# 训练模型时自动进行特征预处理
model.train_model(
    preprocess_features=True,  # 启用自动特征预处理
    correlation_threshold=0.6  # 设置相关性阈值
)
```

## 参数说明

### preprocess_features 方法参数

- `threshold` (float, 默认0.6): 相关性阈值，超过此值的特征对将被视为高度相关
- `plot_correlation_matrix` (bool, 默认True): 是否绘制保留特征的相关矩阵图

### train_model 方法新增参数

- `preprocess_features` (bool, 默认True): 是否在训练前进行特征预处理
- `correlation_threshold` (float, 默认0.6): 特征预处理的相关性阈值

## 功能特性

### 1. 自动特征选择
- 基于相关性阈值自动识别高度相关的特征组
- 在每个相关组中保留数据质量最好的特征（缺失值最少）
- 自动移除冗余特征

### 2. 可视化支持
- 显示原始特征的相关性矩阵
- 显示预处理后特征的相关性矩阵
- 提供详细的预处理摘要信息

### 3. 信息保存
- 自动保存特征预处理信息到模型文件
- 加载模型时自动恢复预处理信息
- 支持查看预处理历史和结果

### 4. 灵活配置
- 可调节相关性阈值
- 支持手动和自动两种预处理模式
- 可选择是否显示可视化图表

## 输出信息

### 预处理摘要示例
```
特征预处理摘要:
==================================================
原始特征数量: 84
保留特征数量: 45
移除特征数量: 39
相关性阈值: 0.6
特征减少比例: 46.4%

移除的特征 (前10个): ['feature1', 'feature2', 'feature3', ...]
  ... 还有 29 个特征

保留的特征 (前10个): ['feature4', 'feature5', 'feature6', ...]
  ... 还有 35 个特征
```

## 优势

1. **减少多重共线性**：通过移除高度相关的特征，减少模型的多重共线性问题
2. **提高数据质量**：优先保留缺失值较少的特征，提高数据质量
3. **降低模型复杂度**：减少特征数量，降低模型复杂度
4. **提高训练效率**：减少特征数量，加快模型训练速度
5. **减少过拟合**：通过特征选择，减少过拟合风险
6. **保持可解释性**：保留最重要的特征，提高模型可解释性

## 注意事项

1. **相关性阈值选择**：阈值过低可能保留过多冗余特征，阈值过高可能移除有用特征
2. **数据质量**：算法假设缺失值较少的特征质量更好，需要根据实际情况调整
3. **特征重要性**：算法主要基于相关性和缺失值，未考虑特征对目标变量的重要性
4. **计算成本**：对于大量特征，计算相关性矩阵可能较耗时

## 示例文件

- `example_feature_preprocessing.py`: 完整的使用示例
- `test_feature_preprocessing.py`: 功能测试脚本

## 版本历史

- v1.0: 初始版本，实现基本的特征预处理功能
- 支持基于相关性和缺失值的特征选择
- 支持手动和自动两种预处理模式
- 支持预处理信息的保存和加载 