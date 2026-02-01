# 日志系统使用指南

## 概述

系统使用 `loguru` 实现统一的日志管理，支持配置驱动、自动轮转、结构化日志和线程安全。

## 快速开始

### 基本使用

```python
from src.utils.logging_config import get_logger

# 获取logger实例
logger = get_logger(__name__)

# 记录日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### 应用启动时初始化

```python
from src.utils.logging_config import setup_logging

# 在应用启动时调用一次
setup_logging()
```

## 配置

日志配置在 `config/system_config.yaml` 中:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
  rotation: "100 MB"  # 文件大小达到100MB时轮转
  retention: "30 days"  # 保留30天的日志
  
  # 控制台输出
  console:
    enabled: true
    colorize: true
  
  # 文件输出
  file:
    enabled: true
    path: "logs/quant_{time:YYYY-MM-DD}.log"
```

## 结构化日志

### 性能日志

```python
from src.utils.logging_config import log_performance

log_performance("backtest_execution", 12.345,
               symbols=100,
               bars_processed=24200,
               memory_mb=450.5)
```

输出:
```
PERF: backtest_execution completed in 12.345s
```

### 交易日志

```python
from src.utils.logging_config import log_trade

log_trade("BUY", "sh.600000", 1000, 15.23,
         commission=1.52,
         strategy="JSG")
```

输出:
```
TRADE: BUY 1000 sh.600000 @ 15.23
```

### 错误日志

```python
from src.utils.logging_config import log_error_with_context

try:
    # 业务逻辑
    result = risky_operation()
except Exception as e:
    log_error_with_context(e, {
        'operation': 'calculate_returns',
        'symbol': 'sh.600000',
        'portfolio_value': 1000000.0
    })
```

## Logger命名

为不同模块使用不同的logger名称:

```python
# 策略模块
strategy_logger = get_logger("JSGStrategy")
strategy_logger.info("Strategy initialized")

# Broker模块
broker_logger = get_logger("BacktestBroker")
broker_logger.info("Broker ready")

# 引擎模块
engine_logger = get_logger("TradingEngine")
engine_logger.info("Engine starting")
```

## 日志文件

### 位置

日志文件保存在 `logs/` 目录:
```
logs/
├── quant_2026-02-01.log
├── quant_2026-02-01.log.zip  # 轮转后压缩
└── quant_2026-02-02.log
```

### 轮转策略

- **大小轮转**: 文件达到100MB时自动轮转
- **时间保留**: 自动删除30天前的日志
- **压缩**: 轮转后的日志自动压缩为zip格式

## 性能

- **吞吐量**: ~5700 条日志/秒
- **线程安全**: 使用 `enqueue=True` 确保多线程安全
- **异步写入**: 不阻塞主线程

## 环境变量覆盖

可以通过环境变量覆盖配置:

```bash
# 设置日志级别
export QUANT_LOGGING_LEVEL=DEBUG

# 禁用文件日志
export QUANT_LOGGING_FILE_ENABLED=false
```

## 最佳实践

### 1. 使用合适的日志级别

- **DEBUG**: 详细的调试信息（开发环境）
- **INFO**: 一般信息（生产环境默认）
- **WARNING**: 警告但不影响运行
- **ERROR**: 错误需要关注

### 2. 包含上下文信息

```python
# ❌ 不好
logger.error("Trade failed")

# ✅ 好
logger.error("Trade failed", 
            symbol="sh.600000",
            quantity=1000,
            reason="insufficient_funds")
```

### 3. 避免敏感信息

```python
# ❌ 不要记录密码、API密钥等
logger.info(f"Connecting with password: {password}")

# ✅ 只记录必要信息
logger.info("Connecting to database", host=host, user=user)
```

### 4. 使用结构化日志

优先使用 `log_performance`, `log_trade` 等结构化日志函数，便于后续分析。

## 故障排查

### 日志文件未创建

1. 检查 `logs/` 目录权限
2. 确认配置中 `file.enabled = true`
3. 查看控制台是否有错误信息

### 日志级别不生效

1. 确认 `config/system_config.yaml` 中的 `level` 设置
2. 检查是否有环境变量覆盖
3. 确保调用了 `setup_logging()`

### 性能问题

如果日志影响性能:
1. 提高日志级别 (INFO → WARNING)
2. 禁用文件日志（仅开发时）
3. 增加轮转大小减少IO

## 示例

完整示例见 `test_logging.py`:

```bash
python test_logging.py
```

## 相关文件

- 配置: [`config/system_config.yaml`](file:///home/zs/workspace/exp/quent/config/system_config.yaml)
- 实现: [`src/utils/logging_config.py`](file:///home/zs/workspace/exp/quent/src/utils/logging_config.py)
- 测试: [`test_logging.py`](file:///home/zs/workspace/exp/quent/test_logging.py)
