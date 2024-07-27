# 因子ML策略回测使用指南

## 概述

本指南介绍如何使用修改后的 `factor_ml_policy.py` 进行回测，支持 jointdata 特征和 LGB 模型。

## 主要功能

### ✅ 已实现的功能

1. **jointdata 支持**: 从 ClickHouse 数据库加载预计算因子
2. **LGB 模型支持**: 加载和使用训练好的 LightGBM 模型
3. **多因子源支持**: 支持 jointdata 因子和 alpha 因子
4. **智能缓存**: 因子数据和预测结果缓存
5. **灵活配置**: 可配置的阈值和参数

### 🔧 核心改进

1. **代码简洁**: 重构了代码结构，提高可读性
2. **错误处理**: 增强了错误处理和日志记录
3. **类型安全**: 添加了类型注解和空值检查
4. **性能优化**: 实现了智能缓存机制

## 使用方法

### 1. 基本配置

```python
from policy.factor_ml_policy import Agent, OrderPolicy
from db import DB

# 初始化数据库客户端
db_client = DB()

# 策略配置
strategy_config = {
    "db_client": db_client,
    "model_path": "./simple_models",
    "model_name": "lgb_model_20241201.txt",  # LGB模型文件
    "use_simple_model": True,
    "use_jointdata": True,  # 启用 jointdata 支持
    "jointdata_config": {
        'host': 'localhost',
        'port': 8123,
        'user': 'default',
        'password': '',
        'database': 'factor_db',
        'table_name': 'factor_data'
    },
    "prediction_threshold": 0.6,  # 买入阈值
    "sell_threshold": 0.4,        # 卖出阈值
    "max_stocks": 10,             # 最大持股数量
}
```

### 2. 使用 jointdata 特征

```python
# 启用 jointdata 支持
strategy_config["use_jointdata"] = True
strategy_config["jointdata_config"] = {
    'host': 'localhost',
    'port': 8123,
    'user': 'default',
    'password': '',
    'database': 'factor_db',
    'table_name': 'factor_data'
}

# 创建代理
agent = Agent(market_env, **strategy_config)
```

### 3. 使用 LGB 模型

```python
# 使用 LGB 模型文件
strategy_config["model_name"] = "lgb_model_20241201.txt"

# 创建代理（会自动加载 LGB 模型）
agent = Agent(market_env, **strategy_config)
```

### 4. 使用 alpha 因子（备用方案）

```python
# 禁用 jointdata，使用 alpha 因子
strategy_config["use_jointdata"] = False
strategy_config["model_name"] = "simple_factor_model_20250608_183915"

# 创建代理
agent = Agent(market_env, **strategy_config)
```

## 配置参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_jointdata` | bool | False | 是否使用 jointdata 特征 |
| `use_simple_model` | bool | True | 是否使用简化模型 |
| `model_name` | str | - | 模型文件名 |
| `prediction_threshold` | float | 0.15 | 买入预测阈值 |
| `sell_threshold` | float | 0.4 | 卖出预测阈值 |
| `max_stocks` | int | 1 | 最大持股数量 |

### jointdata 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | localhost | ClickHouse 主机 |
| `port` | int | 8123 | ClickHouse 端口 |
| `user` | str | default | 用户名 |
| `password` | str | '' | 密码 |
| `database` | str | factor_db | 数据库名 |
| `table_name` | str | factor_data | 表名 |

## 回测流程

### 1. 初始化阶段

```python
# 创建市场环境
market_env = MultiMarketEnv(
    db_client=db_client,
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=1000000,
    commission_rate=0.0003,
    slippage=0.002
)

# 创建策略代理
agent = Agent(market_env, **strategy_config)

# 创建订单策略
order_policy = OrderPolicy(market_env.account, **strategy_config)
```

### 2. 回测循环

```python
# 示例回测循环
for date in trading_dates:
    # 更新当前日期
    agent.current_date = date
    order_policy.current_date = date
    
    # 执行策略决策
    agent.action_decider(market_data)
    
    # 执行订单
    market_env.execute_orders()
    
    # 记录结果
    market_env.record_daily_result()
```

### 3. 结果分析

```python
# 获取账户信息
account = market_env.account

# 计算收益率
initial_value = account.initial_capital
final_value = account.get_total_value()
total_return = (final_value - initial_value) / initial_value * 100

print(f"总收益率: {total_return:.2f}%")
```

## 性能优化

### 1. 缓存机制

- **因子缓存**: 避免重复计算因子
- **预测缓存**: 避免重复预测
- **智能失效**: 基于日期和股票池的缓存失效

### 2. 批量处理

- **批量因子计算**: 一次性计算多只股票的因子
- **批量预测**: 一次性预测多只股票的概率
- **批量订单**: 批量处理交易订单

### 3. 内存管理

- **限制因子数量**: 默认使用前50个因子
- **定期清理**: 自动清理过期缓存
- **内存监控**: 监控内存使用情况

## 错误处理

### 1. 常见错误

- **jointdata 连接失败**: 自动回退到 alpha 因子
- **模型加载失败**: 记录错误并跳过预测
- **因子计算失败**: 使用默认值或跳过

### 2. 日志记录

```python
# 启用详细日志
logger.info("策略初始化完成")
logger.warning("jointdata 连接失败，使用备用方案")
logger.error("模型预测失败")
```

## 示例代码

### 完整回测示例

```python
#!/usr/bin/env python3
"""
完整回测示例
"""

from policy.factor_ml_policy import Agent, OrderPolicy
from db import DB

def run_backtest():
    # 配置
    db_client = DB()
    strategy_config = {
        "db_client": db_client,
        "model_path": "./simple_models",
        "model_name": "lgb_model_20241201.txt",
        "use_simple_model": True,
        "use_jointdata": True,
        "jointdata_config": {
            'host': 'localhost',
            'port': 8123,
            'user': 'default',
            'password': '',
            'database': 'factor_db',
            'table_name': 'factor_data'
        },
        "prediction_threshold": 0.6,
        "sell_threshold": 0.4,
        "max_stocks": 10,
    }
    
    # 创建环境
    market_env = MultiMarketEnv(
        db_client=db_client,
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=1000000,
        commission_rate=0.0003,
        slippage=0.002
    )
    
    # 创建代理
    agent = Agent(market_env, **strategy_config)
    
    # 执行回测
    # ... 回测逻辑
    
    print("回测完成")

if __name__ == "__main__":
    run_backtest()
```

## 注意事项

1. **依赖安装**: 确保安装了 `clickhouse-connect` 和 `lightgbm`
2. **数据准备**: 确保 jointdata 数据库中有相应的因子数据
3. **模型文件**: 确保 LGB 模型文件存在且格式正确
4. **内存使用**: 监控内存使用，避免内存溢出
5. **网络连接**: 确保能够连接到 ClickHouse 数据库

## 故障排除

### 1. jointdata 连接失败

```python
# 检查连接
try:
    from jointdata.wide_table_manager import WideTableManager
    manager = WideTableManager(host='localhost', port=8123)
    print("连接成功")
except Exception as e:
    print(f"连接失败: {e}")
```

### 2. 模型加载失败

```python
# 检查模型文件
import os
model_file = "./simple_models/lgb_model_20241201.txt"
if os.path.exists(model_file):
    print("模型文件存在")
else:
    print("模型文件不存在")
```

### 3. 因子数据缺失

```python
# 检查因子数据
factors = manager.get_available_factors()
print(f"可用因子数量: {len(factors)}")
```

## 总结

修改后的 `factor_ml_policy.py` 提供了：

1. **简洁高效的代码结构**
2. **完整的 jointdata 支持**
3. **灵活的 LGB 模型集成**
4. **智能的缓存机制**
5. **完善的错误处理**

通过这些改进，可以更高效地进行因子策略回测，同时保持良好的代码可维护性。 