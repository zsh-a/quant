# 因子模型二分类修改总结

## 概述
将 `SimpleFactorModel` 从多分类问题修改为二分类问题，使用未来收益与股池平均收益对比的方式计算标签。

## 主要修改

### 1. 模型配置修改
- 将默认的 `n_groups` 从 5 改为 2
- 更新注释说明为"分组数量（2组：高收益vs低收益）"

### 2. 标签计算逻辑修改
**修改前**: 使用中位数进行分组，高于中位数为1，低于中位数为0
**修改后**: 使用平均收益进行分组，大于等于平均收益为1，否则为0

#### 修改的函数:
- `_calculate_labels()` - 主要标签计算函数
- `assign_binary_labels_fast()` - 快速二分类标签计算
- `assign_binary_labels()` - 备用二分类标签计算

### 3. 模型训练参数修改
- 将 `objective` 从 `"multiclass"` 改为 `"binary"`
- 移除了 `num_class` 参数（二分类不需要）

### 4. 评估逻辑修改
- 更新了评估函数中的文本和逻辑
- 将返回结果中的 `"high_return_group"` 改为 `"above_average_group"`
- 更新了分类报告中的标签名称

### 5. 文本和注释更新
- 更新了所有相关的打印文本和注释
- 将"高收益组"改为"高于等于平均收益组"
- 将"低收益组"改为"低于平均收益组"

## 标签计算逻辑详解

### 新的标签计算方式:
1. 对于每个交易日，计算该日股池中所有股票的未来收益率
2. 计算该日股池的平均收益率
3. 对于每只股票：
   - 如果未来收益率 >= 平均收益率，标签为 1
   - 如果未来收益率 < 平均收益率，标签为 0

### 优势:
- 更直观的标签含义：1表示表现优于平均，0表示表现低于平均
- 每个交易日的标签分布更加平衡
- 避免了极端值对分组的影响

## 使用示例

```python
# 创建二分类模型
model = SimpleFactorModel(
    train_start="20220101",
    train_end="20221231", 
    test_start="20230101",
    test_end="20231231",
    n_groups=2,  # 二分类
    future_days=5,
    data_frequency="D",
    use_jointdata=False,
)

# 准备数据
model.load_and_prepare_data()

# 训练模型
model.train_model()

# 评估模型
results = model.evaluate_model()

# 获取关键指标
print(f"测试集准确率: {results['overall']['test_accuracy']:.4f}")
print(f"测试集ROC AUC: {results['above_average_group']['test']['roc_auc']:.4f}")
print(f"测试集F1 Score: {results['above_average_group']['test']['f1']:.4f}")
```

## 注意事项

1. **数据要求**: 确保每个交易日有足够的股票数据来计算有意义的平均收益
2. **标签分布**: 理论上每个交易日的标签分布应该相对平衡（接近50%-50%）
3. **模型解释**: 模型预测的是股票未来表现是否优于股池平均水平的概率

## 兼容性

- 保持了原有的API接口
- 所有原有的方法都可以正常使用
- 只是改变了内部的标签计算逻辑和相关的文本描述 