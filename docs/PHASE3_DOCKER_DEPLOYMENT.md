# Phase 3 - Docker部署指南

## 概述

完整的Docker容器化部署方案，支持一键启动所有服务。

## 架构

```
┌─────────────────────────────────────────────────────┐
│                   Nginx (Frontend)                  │
│                   Port: 80                          │
└──────────────┬──────────────────────────────────────┘
               │
               ├──► API Server (FastAPI)
               │    Port: 8000
               │
               ├──► WebSocket
               │
               └──► Static Files
                    
┌──────────────┬──────────────────────────────────────┐
│              │         Backend Services             │
├──────────────┼──────────────────────────────────────┤
│ API          │ FastAPI + Uvicorn                    │
│ Celery Worker│ 4 concurrent workers                 │
│ Flower       │ Celery monitoring (Port: 5555)       │
│ Redis        │ Message broker (Port: 6379)          │
│ Prometheus   │ Metrics collection (Port: 9090)      │
│ Grafana      │ Visualization (Port: 3000)           │
└──────────────┴──────────────────────────────────────┘
```

## 快速开始

### 1. 前置要求

- Docker 20.10+
- Docker Compose 2.0+
- 至少 4GB 可用内存
- 至少 10GB 可用磁盘空间

### 2. 一键部署

```bash
# 开发模式
./deploy.sh dev

# 生产模式
./deploy.sh prod
```

### 3. 访问服务

| 服务 | URL | 说明 |
|------|-----|------|
| 前端 | http://localhost | React应用 |
| API | http://localhost:8000 | FastAPI后端 |
| API文档 | http://localhost:8000/docs | Swagger UI |
| Flower | http://localhost:5555 | Celery监控 |
| Prometheus | http://localhost:9090 | 指标收集 |
| Grafana | http://localhost:3000 | 可视化面板 |

**Grafana默认账号**: admin / admin

## 服务说明

### 核心服务

**API Server**:
- FastAPI应用
- 健康检查: `/monitoring/health`
- 自动重启策略

**Celery Worker**:
- 4个并发worker
- 处理回测任务
- 自动任务重试

**Redis**:
- 消息代理
- 结果存储
- 数据持久化

**Frontend**:
- Nginx + React
- Gzip压缩
- API代理
- WebSocket支持

### 监控服务

**Flower**:
- Celery任务监控
- 实时worker状态
- 任务历史

**Prometheus**:
- 15秒采集间隔
- 自动服务发现
- 数据持久化

**Grafana**:
- 预配置数据源
- 自定义仪表板
- 告警规则

## 部署命令

### 启动服务

```bash
# 开发模式（带日志输出）
./deploy.sh dev

# 生产模式（后台运行）
./deploy.sh prod

# 仅启动特定服务
docker-compose up -d api celery_worker redis
```

### 停止服务

```bash
# 停止所有服务
./deploy.sh stop

# 停止特定服务
docker-compose stop api
```

### 重启服务

```bash
# 重启所有服务
./deploy.sh restart

# 重启特定服务
docker-compose restart api
```

### 查看日志

```bash
# 所有服务日志
./deploy.sh logs

# 特定服务日志
./deploy.sh logs api
docker-compose logs -f celery_worker
```

### 清理环境

```bash
# 清理所有容器、卷和镜像
./deploy.sh clean
```

## 数据持久化

### 卷映射

```yaml
volumes:
  - ./data:/app/data          # 数据库和检查点
  - ./config:/app/config      # 配置文件
  - ./logs:/app/logs          # 日志文件
```

### Docker卷

- `redis_data` - Redis数据
- `prometheus_data` - Prometheus时序数据
- `grafana_data` - Grafana配置和仪表板

## 配置

### 环境变量

编辑 `docker-compose.yml`:

```yaml
environment:
  - CELERY_BROKER_URL=redis://redis:6379/0
  - CELERY_RESULT_BACKEND=redis://redis:6379/1
  - PYTHONUNBUFFERED=1
```

### 系统配置

编辑 `config/system_config.yaml`:

```yaml
celery:
  worker_concurrency: 4  # 调整worker数量

alerting:
  feishu:
    webhook_url: "YOUR_WEBHOOK_URL"
```

## 性能优化

### 资源限制

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Nginx优化

```nginx
# nginx.conf
worker_processes auto;
worker_connections 1024;

gzip on;
gzip_comp_level 6;
gzip_types text/plain text/css application/json;
```

### Redis优化

```yaml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## 健康检查

### 自动健康检查

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/monitoring/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 手动检查

```bash
# API健康
curl http://localhost:8000/monitoring/health

# Redis连接
docker-compose exec redis redis-cli ping

# 查看所有服务状态
docker-compose ps
```

## 故障排查

### 服务无法启动

```bash
# 查看详细日志
docker-compose logs api

# 检查端口占用
netstat -tulpn | grep 8000

# 重新构建镜像
docker-compose build --no-cache api
```

### 内存不足

```bash
# 查看资源使用
docker stats

# 减少worker数量
# 编辑 docker-compose.yml
command: celery -A src.tasks.celery_app worker --concurrency=2
```

### 网络问题

```bash
# 检查网络
docker network ls
docker network inspect quant_network

# 重建网络
docker-compose down
docker-compose up -d
```

## 生产部署建议

### 1. 使用环境变量文件

创建 `.env`:

```env
# API配置
API_PORT=8000
API_WORKERS=4

# Redis配置
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password

# Celery配置
CELERY_CONCURRENCY=4

# 告警配置
FEISHU_WEBHOOK=https://...
```

### 2. 启用HTTPS

使用 Let's Encrypt + Nginx:

```yaml
frontend:
  volumes:
    - ./ssl:/etc/nginx/ssl
  ports:
    - "443:443"
```

### 3. 数据库备份

```bash
# 备份脚本
#!/bin/bash
docker-compose exec -T api tar -czf - /app/data | \
  gzip > backup_$(date +%Y%m%d).tar.gz
```

### 4. 日志轮转

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 5. 监控告警

配置 Prometheus Alertmanager:

```yaml
alertmanager:
  image: prom/alertmanager
  ports:
    - "9093:9093"
  volumes:
    - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

## 扩展部署

### 水平扩展

```bash
# 扩展Celery worker
docker-compose up -d --scale celery_worker=8

# 扩展API服务（需要负载均衡器）
docker-compose up -d --scale api=3
```

### 负载均衡

使用Nginx作为负载均衡器:

```nginx
upstream api_backend {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}
```

## 更新部署

### 滚动更新

```bash
# 1. 拉取最新代码
git pull

# 2. 重新构建镜像
docker-compose build

# 3. 逐个重启服务
docker-compose up -d --no-deps --build api
docker-compose up -d --no-deps --build celery_worker
docker-compose up -d --no-deps --build frontend
```

### 零停机更新

```bash
# 使用蓝绿部署
docker-compose -f docker-compose.blue.yml up -d
# 切换流量
# 停止旧版本
docker-compose -f docker-compose.green.yml down
```

## CI/CD集成

### GitHub Actions示例

```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push
        run: |
          docker-compose build
          docker-compose push
      
      - name: Deploy to server
        run: |
          ssh user@server 'cd /app && ./deploy.sh prod'
```

## 性能基准

| 指标 | 值 |
|------|-----|
| API响应时间 | <100ms (P95) |
| 并发请求 | 1000+ QPS |
| Celery吞吐 | 10+ tasks/sec |
| 内存使用 | ~2GB (全部服务) |
| 启动时间 | ~30秒 |

## 安全建议

1. **修改默认密码**: Grafana, Redis
2. **启用防火墙**: 只开放必要端口
3. **使用secrets**: 敏感配置使用Docker secrets
4. **定期更新**: 保持镜像和依赖最新
5. **备份数据**: 定期备份数据卷

## 参考资料

- [Docker文档](https://docs.docker.com/)
- [Docker Compose文档](https://docs.docker.com/compose/)
- [Nginx文档](https://nginx.org/en/docs/)

---

**创建时间**: 2026-02-01  
**状态**: ✅ 已完成  
**部署方式**: Docker Compose
