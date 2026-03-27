# 量化交易平台 - 快速启动指南

## 🚀 快速开始

### 1. 系统健康检查

```bash
python health_check.py
```

预期输出：`7/7 checks passed (100%)` ✅

### 2. 本地快速调试（推荐）

```bash
./dev_local.sh up
```

这会自动：
- 优先复用本机 `redis://127.0.0.1:6379`，没有则自动启动本地 Redis 或 `docker compose` 里的 `redis`
- 本地启动 API（热重载）、Celery worker（包含 `backtest/default/automation` 队列）和前端 Vite
- 日志写入 `.dev/local/logs/`

常用命令：

```bash
./dev_local.sh status
./dev_local.sh logs api
./dev_local.sh logs worker
./dev_local.sh down
```

也可以使用：

```bash
make dev_local
make dev_local_status
make dev_local_down
```

### 3. Docker 单机部署

```bash
./deploy.sh up
```

常用命令：

```bash
./deploy.sh build
./deploy.sh status
./deploy.sh logs
./deploy.sh logs api
./deploy.sh restart
./deploy.sh down
```

默认 Docker 部署只包含核心服务：
- `frontend`
- `api`
- `redis`
- `celery_worker`

说明：
- `celery_worker` 是异步回测和自动化任务执行器，属于核心服务
- `ClickHouse` 仍按外部依赖处理，不由当前 `docker-compose.yml` 启动

### 4. 手动启动后端API

```bash
uvicorn src.api.server:app --reload
```

访问 http://localhost:8000/docs 查看API文档

### 5. 启动前端（新终端）

```bash
cd ui && npm run dev
```

访问 http://localhost:5173

### 6. 本地触发加密分钟数据初始化/同步

无需先起 API，可直接本地执行：

```bash
python -m src.market_data.crypto_cli init-db
python -m src.market_data.crypto_cli bootstrap --provider bitget --symbols BTCUSDT,ETHUSDT
python -m src.market_data.crypto_cli backfill --provider bitget --symbols BTCUSDT,ETHUSDT --start 2020-01-01T00:00:00+00:00
python -m src.market_data.crypto_cli sync --provider bitget --symbols BTCUSDT --interval 1m --start 2026-03-27T00:00:00+00:00 --end 2026-03-28T00:00:00+00:00
python -m src.market_data.crypto_cli overview
python -m src.market_data.crypto_cli coverage --interval 1m --limit 20
```

---

## 🧪 运行测试

```bash
# 性能基准
python benchmark_datastream.py

# 实时数据流
python test_realtime_stream.py

# 日志系统
python test_logging.py

# E2E集成（需要API服务器运行）
python test_e2e_integration.py
```

---

## 📁 项目结构

```
quent/
├── src/
│   ├── api/              # API服务器
│   ├── core/             # 核心组件
│   ├── strategies/       # 交易策略
│   └── utils/            # 工具（配置、日志）
├── ui/                   # React前端
├── config/               # 配置文件
├── docs/                 # 文档
└── logs/                 # 日志（自动创建）
```

---

## 🔧 配置

编辑 `config/system_config.yaml`:

```yaml
api:
  port: 8000

logging:
  level: "INFO"
  rotation: "100 MB"
  retention: "30 days"

data_stream:
  chunk_size_months: null  # 自动
```

环境变量覆盖:
```bash
export QUANT_API_PORT=8080
export QUANT_LOGGING_LEVEL=DEBUG
```

---

## 📚 文档

- [日志系统](docs/LOGGING.md)
- [E2E测试](docs/E2E_TESTING.md)
- [完整实施总结](brain/walkthrough.md)

---

## 🐛 故障排查

### API服务器启动失败
```bash
# 检查端口占用
lsof -i :8000
```

### 数据库连接失败
1. 确认ClickHouse运行中
2. 检查配置文件
3. 验证网络连接

### 查看日志
```bash
tail -f logs/quant_$(date +%Y-%m-%d).log
```

---

**版本**: Phase 1 ✅  
**状态**: 生产就绪  
**最后更新**: 2026-02-01
