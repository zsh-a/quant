# Phase 3 - 并发回测系统

## 概述

Phase 3实现了基于Celery的分布式任务队列系统，支持多策略并行回测。

## 架构

```
API Server (FastAPI)
    ↓
Redis (Message Broker)
    ↓
Celery Workers (4 concurrent)
    ↓
Backtest Tasks
```

## 快速开始

### 1. 启动Redis

```bash
# 使用Docker
docker run -d -p 6379:6379 redis:7-alpine

# 或使用系统Redis
redis-server
```

### 2. 启动Celery Worker

```bash
./start_worker.sh
```

或手动启动：

```bash
celery -A src.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=backtest,default
```

### 3. 启动API服务器

```bash
uvicorn src.api.server:app --reload --port 8000
```

### 4. 提交回测任务

```bash
curl -X POST http://localhost:8000/tasks/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_001",
    "symbol": "000001.SZ",
    "strategy": "jsg",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "params": {},
    "initial_cash": 1000000,
    "enable_risk_management": true
  }'
```

响应：
```json
{
  "task_id": "abc123...",
  "session_id": "test_001",
  "status": "submitted"
}
```

### 5. 查询任务状态

```bash
curl http://localhost:8000/tasks/backtest/{task_id}
```

响应：
```json
{
  "task_id": "abc123...",
  "status": "PROGRESS",
  "progress": 45.2,
  "message": "Processing bar 452/1000"
}
```

## API端点

### 任务管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/tasks/backtest` | POST | 提交回测任务 |
| `/tasks/backtest/{task_id}` | GET | 查询任务状态 |
| `/tasks/backtest/{task_id}` | DELETE | 取消任务 |
| `/tasks/backtest` | GET | 列出所有任务 |
| `/tasks/workers` | GET | 查看Worker状态 |

### 任务状态

- `PENDING`: 等待执行
- `PROGRESS`: 执行中
- `SUCCESS`: 成功完成
- `FAILURE`: 执行失败

## 监控

### Flower (Celery监控面板)

```bash
celery -A src.tasks.celery_app flower --port=5555
```

访问: http://localhost:5555

### 查看Worker状态

```bash
curl http://localhost:8000/tasks/workers
```

## 配置

配置文件: `config/system_config.yaml`

```yaml
celery:
  broker_url: "redis://localhost:6379/0"
  result_backend: "redis://localhost:6379/1"
  worker_concurrency: 4
  task_time_limit: 3600
```

## 性能优化

### 并发数调整

根据CPU核心数调整worker并发数：

```bash
celery -A src.tasks.celery_app worker --concurrency=8
```

### 内存管理

Worker会在执行100个任务后自动重启，防止内存泄漏：

```yaml
max_tasks_per_child: 100
```

### 任务优先级

高优先级任务：

```python
task.apply_async(args=[...], priority=9)
```

## 测试

运行测试脚本：

```bash
python test_celery_tasks.py
```

测试内容：
1. Worker状态检查
2. 任务提交
3. 进度监控
4. 结果获取

## 故障排查

### Redis连接失败

```bash
# 检查Redis是否运行
redis-cli ping
# 应返回: PONG
```

### Worker无法启动

```bash
# 检查依赖
pip install celery redis

# 查看详细日志
celery -A src.tasks.celery_app worker --loglevel=debug
```

### 任务一直PENDING

- 检查Worker是否运行
- 检查队列名称是否正确
- 查看Flower监控面板

## 下一步

- [ ] 前端集成（显示任务进度）
- [ ] 任务优先级队列
- [ ] 动态Worker扩容
- [ ] 任务结果缓存

## 参考资料

- [Celery文档](https://docs.celeryq.dev/)
- [Redis文档](https://redis.io/docs/)
- [Flower文档](https://flower.readthedocs.io/)
