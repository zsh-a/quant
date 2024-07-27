# Factor ML Policy 二分类修改总结

## ✅ 修改完成状态

**`policy/factor_ml_policy.py` 已成功更新为使用二分类模型**

## 🔄 核心修改内容

### 1. 类和方法注释更新
- **Agent类**: 从"基于因子模型的智能交易代理"改为"基于二分类因子模型的智能交易代理"
- **预测函数**: 明确说明预测的是"股票属于高于等于平均收益组的概率"
- **决策函数**: 更新注释以反映二分类模型的特性

### 2. 预测逻辑说明
- **预测含义**: 模型预测的是股票未来表现是否优于股池平均水平的概率
- **买入条件**: 当预测概率 > `prediction_threshold` 时买入
- **卖出条件**: 当预测概率 < `sell_threshold` 时卖出

### 3. 日志信息优化
- **买入决策**: 显示"高于平均收益概率"而不是"预测概率"
- **卖出决策**: 显示"高于平均收益概率"而不是"预测概率"
- **选股结果**: 明确标注选中的是"高于平均收益概率较高"的股票

## 📊 修改的具体位置

### 1. 类定义更新
```python
class Agent:
    """基于二分类因子模型的智能交易代理 - 负责所有下单决策逻辑"""
```

### 2. 预测函数更新
```python
def predict_stock_probabilities(self, stocks, current_date):
    """批量预测股票属于高于等于平均收益组的概率"""
    
    # 进行二分类预测（预测股票属于高于等于平均收益组的概率）
    predictions = self.simple_factor_model.predict_highest_group_proba(X_pred)
    
    logger.info(f"成功预测 {len(result)} 只股票属于高于等于平均收益组的概率")
```

### 3. 买入决策逻辑更新
```python
def should_buy(self, code, probability):
    """买入决策逻辑 - 基于二分类模型预测"""
    
    # 检查预测概率是否超过阈值（预测股票属于高于等于平均收益组的概率）
    buy_signal = probability > self.prediction_threshold
    
    if buy_signal:
        logger.info(
            f"买入决策: {code}, 高于平均收益概率: {probability:.4f}, 阈值: {self.prediction_threshold}"
        )
```

### 4. 卖出决策逻辑更新
```python
def should_sell(self, code, probability):
    """卖出决策逻辑 - 基于二分类模型预测"""
    
    # 如果预测概率较低，则卖出（预测股票属于高于等于平均收益组的概率较低）
    sell_signal = probability < self.sell_threshold
    
    if sell_signal:
        logger.info(
            f"卖出决策: {code}, 高于平均收益概率: {probability:.4f}, 阈值: {self.sell_threshold}"
        )
```

### 5. 交易决策函数更新
```python
def action_decider(self, stocks_obs):
    """交易决策 - 基于二分类模型预测进行每日调仓"""
    
    # 选择高预测概率的股票（预测属于高于等于平均收益组概率较高的股票）
    high_prob_stocks = [...]
    
    logger.info(f"选中目标股票（高于平均收益概率较高）: {target_stocks}")
```

### 6. 月度再平衡函数更新
```python
def monthly_rebalance(self):
    """月度再平衡 - 基于二分类模型预测"""
    
    # 选择高预测概率的股票（预测属于高于等于平均收益组概率较高的股票）
    high_prob_stocks = [...]
    
    logger.info(f"月度再平衡 - 选中目标股票（高于平均收益概率较高）: {target_stocks}")
    for code, prob in high_prob_stocks[: self.max_stocks]:
        logger.info(f"  {code}: 高于平均收益概率 {prob:.4f}")
```

## 🎯 二分类模型的优势

1. **更直观的决策**: 直接预测股票是否优于平均水平
2. **更清晰的阈值**: 0.5作为理论上的中性阈值
3. **更平衡的分布**: 每个交易日的标签分布更接近50%-50%
4. **更实用的解释**: 符合实际投资决策的需求

## 📋 使用建议

### 阈值设置
- **买入阈值 (`prediction_threshold`)**: 建议设置为 0.5-0.7
  - 0.5: 中性阈值，预测优于平均就买入
  - 0.6: 保守阈值，预测明显优于平均才买入
  - 0.7: 严格阈值，预测显著优于平均才买入

- **卖出阈值 (`sell_threshold`)**: 建议设置为 0.3-0.5
  - 0.3: 宽松阈值，预测明显低于平均才卖出
  - 0.4: 中性阈值，预测低于平均就卖出
  - 0.5: 严格阈值，预测不优于平均就卖出

### 参数配置示例
```python
args = {
    "prediction_threshold": 0.6,  # 买入阈值
    "sell_threshold": 0.4,        # 卖出阈值
    "max_stocks": 6,              # 最大持股数量
    "model_path": "./simple_models",
    "model_name": "simple_factor_model_20250608_183915",
    "use_simple_model": True,
}
```

## ✅ 验证结果

- ✅ 代码编译通过
- ✅ 语法正确
- ✅ 所有API接口保持兼容
- ✅ 预测逻辑正确更新
- ✅ 日志信息清晰明确
- ✅ 注释准确反映二分类特性

## 🎉 总结

`factor_ml_policy.py` 已成功更新为使用二分类模型，所有相关的注释、日志和逻辑都已相应调整。现在策略会基于模型预测的"股票是否优于平均水平"的概率来进行交易决策，这使得策略更加直观和实用。 