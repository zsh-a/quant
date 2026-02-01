# Phase 3 - 监控告警系统

## 概述

实现了完整的监控告警系统，包括Prometheus指标收集、健康检查、系统状态监控和飞书通知集成。

## 架构

```
Application
    ↓
Metrics Collection (Prometheus)
    ↓
Health Checks (CPU/Memory/Disk/DB/Redis)
    ↓
Alert Rules Engine
    ↓
Feishu Webhook Notifications
```

## 核心功能

### 1. Prometheus指标 ✅

**指标类型**:

**API指标**:
- `api_requests_total` - 总请求数（按方法、端点、状态）
- `api_request_duration_seconds` - 请求耗时

**回测指标**:
- `backtest_duration_seconds` - 回测执行时间
- `backtest_total` - 回测总数（按策略、状态）
- `backtest_trades_count` - 交易数量分布

**系统指标**:
- `system_cpu_usage_percent` - CPU使用率
- `system_memory_usage_bytes` - 内存使用量
- `system_disk_usage_percent` - 磁盘使用率

**会话指标**:
- `active_sessions` - 活跃会话数
- `session_progress_percent` - 会话进度

**Celery指标**:
- `celery_tasks_active` - 活跃任务数
- `celery_tasks_total` - 总任务数
- `celery_task_duration_seconds` - 任务执行时间

### 2. 健康检查 ✅

**检查项**:
- CPU使用率（阈值：80%）
- 内存使用率（阈值：90%）
- 磁盘使用率（阈值：90%）
- 数据库连接
- Redis连接（可选）

**状态级别**:
- `healthy` - 所有检查通过
- `degraded` - 部分检查警告
- `unhealthy` - 关键检查失败

### 3. 告警系统 ✅

**告警规则**:
```yaml
rules:
  - name: "high_cpu"
    condition: "cpu > 80"
    duration: 300      # 持续5分钟
    severity: "warning"
  
  - name: "high_memory"
    condition: "memory > 90"
    duration: 300
    severity: "critical"
```

**告警特性**:
- 条件触发（支持 > 和 < 比较）
- 持续时间检查（避免误报）
- 冷却期（避免告警风暴）
- 多严重级别（info/warning/critical）

### 4. 飞书通知 ✅

**消息格式**:
- 卡片式消息
- 颜色编码（蓝色/橙色/红色）
- 详细信息展示
- 时间戳

## API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/monitoring/health` | GET | 健康检查 |
| `/monitoring/metrics` | GET | Prometheus指标 |
| `/monitoring/status` | GET | 详细系统状态 |
| `/monitoring/alert/test` | POST | 测试告警通知 |

## 使用方法

### 1. 配置飞书Webhook

编辑 `config/system_config.yaml`:

```yaml
alerting:
  enabled: true
  feishu:
    webhook_url: "https://open.feishu.cn/open-apis/bot/v2/hook/YOUR_WEBHOOK_KEY"
```

### 2. 启动服务

```bash
uvicorn src.api.server:app --reload
```

### 3. 查看健康状态

```bash
curl http://localhost:8000/monitoring/health
```

响应：
```json
{
  "status": "healthy",
  "timestamp": "2026-02-01T16:00:00",
  "checks": [
    {
      "component": "cpu",
      "status": "healthy",
      "message": "CPU usage normal: 25.3%",
      "value": 25.3,
      "threshold": 80
    },
    ...
  ]
}
```

### 4. 查看Prometheus指标

```bash
curl http://localhost:8000/monitoring/metrics
```

### 5. 测试告警

```bash
curl -X POST "http://localhost:8000/monitoring/alert/test?title=Test&message=Hello&severity=info"
```

## 集成Grafana（可选）

### 1. 配置Prometheus

`prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'quant_api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/monitoring/metrics'
    scrape_interval: 15s
```

### 2. 启动Prometheus

```bash
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### 3. 启动Grafana

```bash
docker run -d -p 3000:3000 grafana/grafana
```

### 4. 配置数据源

- 访问 http://localhost:3000
- 添加Prometheus数据源: http://localhost:9090
- 导入仪表板

## 告警规则示例

### CPU告警

```python
from src.monitoring.alerts import alert_manager
from src.monitoring.metrics import system_cpu_usage

# 更新CPU指标
cpu_percent = 85.0
system_cpu_usage.set(cpu_percent)

# 检查告警
alert_manager.check_metric('high_cpu', cpu_percent)
```

### 自定义告警

```python
from src.monitoring.alerts import alert_manager, AlertSeverity

alert_manager.send_alert(
    title="Backtest Failed",
    message="Strategy JSG backtest failed with error",
    severity=AlertSeverity.CRITICAL,
    details={
        "strategy": "JSG",
        "error": "Data stream error",
        "session_id": "abc123"
    }
)
```

## 性能影响

| 操作 | 开销 |
|------|------|
| 指标更新 | <1ms |
| 健康检查 | <50ms |
| 告警检查 | <5ms |
| 飞书通知 | <500ms |

## 最佳实践

### 1. 合理设置阈值

```yaml
# 生产环境建议
rules:
  - name: "high_cpu"
    condition: "cpu > 80"
    duration: 300  # 5分钟持续才告警
  
  - name: "critical_memory"
    condition: "memory > 95"
    duration: 60   # 1分钟即告警
```

### 2. 避免告警风暴

```python
# 告警规则自带冷却期
rule = AlertRule(
    name="high_cpu",
    condition=lambda x: x > 80,
    duration=300,
    cooldown=3600  # 1小时内不重复告警
)
```

### 3. 定期更新系统指标

```python
import asyncio
from src.monitoring.metrics import update_system_metrics

async def metric_updater():
    while True:
        update_system_metrics()
        await asyncio.sleep(30)  # 每30秒更新
```

## 故障排查

### 健康检查失败

**问题**: `/monitoring/health` 返回unhealthy
**排查**:
1. 检查具体失败的组件
2. 查看日志: `tail -f logs/app.log`
3. 验证数据库/Redis连接

### 告警未发送

**问题**: 告警规则触发但未收到通知
**排查**:
1. 检查Webhook URL配置
2. 测试告警: `curl -X POST .../alert/test`
3. 查看日志中的错误信息

### Prometheus指标缺失

**问题**: 某些指标未显示
**排查**:
1. 确认指标已注册
2. 检查指标更新逻辑
3. 验证Prometheus配置

## 监控仪表板建议

### 关键指标

**系统健康**:
- CPU使用率趋势
- 内存使用率趋势
- 磁盘使用率趋势

**API性能**:
- 请求QPS
- P95/P99延迟
- 错误率

**回测性能**:
- 平均执行时间
- 成功率
- 并发任务数

## 下一步优化

- [ ] 自定义Grafana仪表板
- [ ] 更多告警渠道（邮件、短信）
- [ ] 告警聚合和去重
- [ ] 历史告警查询

## 依赖

```txt
prometheus-client==0.19.0
psutil==5.9.6
requests==2.31.0
```

## 参考资料

- [Prometheus文档](https://prometheus.io/docs/)
- [飞书机器人文档](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN)
- [Grafana文档](https://grafana.com/docs/)

---

**创建时间**: 2026-02-01  
**状态**: ✅ 已完成  
**覆盖范围**: 指标收集、健康检查、告警通知
