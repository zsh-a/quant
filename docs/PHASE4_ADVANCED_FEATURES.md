# Phase 4 高级特性

## 概述

Phase 4实现了量化交易平台的高级功能，包括多策略组合、参数优化、归因分析和报告生成。

## 功能清单

### 1. 多策略组合 ✅

**核心模块**: `src/portfolio/`

| 文件 | 功能 |
|------|------|
| `portfolio_manager.py` | 多策略管理、权重分配 |
| `backtest.py` | 组合回测引擎 |

**权重分配策略**:
- 等权重 (equal)
- 波动率倒数 (vol_inverse)
- 夏普比率加权 (sharpe)
- 自定义权重 (custom)

**API端点**:
```
POST /portfolio           # 创建组合
GET  /portfolio           # 列出组合
GET  /portfolio/{id}      # 组合详情
PUT  /portfolio/{id}/weights  # 调整权重
POST /portfolio/{id}/backtest # 运行组合回测
```

---

### 2. 参数优化器 ✅

**核心模块**: `src/optimizer/`

| 文件 | 功能 |
|------|------|
| `optimizer.py` | 网格搜索、贝叶斯优化 |

**优化方法**:
- 网格搜索 (grid)
- 随机搜索 (random)
- 贝叶斯优化 (bayesian)

**优化目标**:
- `max_sharpe` - 最大化夏普比率
- `max_return` - 最大化收益率
- `min_drawdown` - 最小化最大回撤
- `max_calmar` - 最大化卡玛比率

**API端点**:
```
POST   /optimize           # 提交优化任务
GET    /optimize/{id}      # 查询状态
GET    /optimize/{id}/results  # 获取结果
GET    /optimize/{id}/heatmap  # 参数热力图
DELETE /optimize/{id}      # 取消任务
```

---

### 3. 归因分析 ✅

**核心模块**: `src/analysis/`

| 文件 | 功能 |
|------|------|
| `attribution.py` | 收益归因、风险归因 |

**归因维度**:
- 按资产 (by_asset)
- 按行业 (by_sector)
- 按时间 (by_period)

**风险指标**:
- 年化波动率
- VaR(95%)
- CVaR(95%)
- 最大回撤

**API端点**:
```
GET /analysis/attribution/{session_id}  # 收益归因
GET /analysis/risk/{session_id}         # 风险分析
GET /analysis/summary/{session_id}      # 快速摘要
```

---

### 4. 报告生成 ✅

**核心模块**: `src/reports/`

| 文件 | 功能 |
|------|------|
| `generator.py` | Markdown报告生成 |

**报告内容**:
- 执行摘要
- 绩效指标
- 交易分析
- 归因分析
- 月度收益

**API端点**:
```
GET /analysis/report/{session_id}  # 生成报告
```

---

## 使用示例

### 创建多策略组合

```python
import requests

# 创建组合
resp = requests.post('http://localhost:8000/portfolio', json={
    'name': 'my_portfolio',
    'strategies': [
        {'name': 'jsg_1', 'strategy': 'jsg', 'params': {}, 'weight': 0.5},
        {'name': 'rotation_1', 'strategy': 'rotation', 'params': {}, 'weight': 0.5}
    ],
    'weight_method': 'equal',
    'rebalance_frequency': 'weekly'
})

portfolio_id = resp.json()['portfolio_id']

# 运行回测
resp = requests.post(f'http://localhost:8000/portfolio/{portfolio_id}/backtest', json={
    'start_date': '2023-01-01',
    'end_date': '2024-01-01',
    'symbols': ['sz.300750', 'sz.002475'],
    'initial_capital': 1000000
})

print(resp.json())
```

### 参数优化

```python
# 提交优化任务
resp = requests.post('http://localhost:8000/optimize', json={
    'strategy': 'jsg',
    'param_space': [
        {'name': 'ma_short', 'param_type': 'int', 'low': 5, 'high': 20, 'step': 1},
        {'name': 'ma_long', 'param_type': 'int', 'low': 20, 'high': 60, 'step': 5}
    ],
    'method': 'bayesian',
    'objective': 'max_sharpe',
    'n_iterations': 50,
    'backtest_config': {
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        'symbols': ['sz.300750']
    }
})

task_id = resp.json()['task_id']

# 查询结果
resp = requests.get(f'http://localhost:8000/optimize/{task_id}')
print(resp.json())
```

### 生成报告

```python
# 获取归因分析
resp = requests.get(f'http://localhost:8000/analysis/attribution/{session_id}')
print(resp.json())

# 生成Markdown报告
resp = requests.get(f'http://localhost:8000/analysis/report/{session_id}')
# 报告保存在 data/reports/ 目录
```

---

## 依赖

```
# pyproject.toml 新增
scikit-learn>=1.3.0  # 贝叶斯优化
scipy>=1.11.0        # 统计计算
```

---

**状态**: ✅ Phase 4 核心功能已完成  
**日期**: 2026-02-01
