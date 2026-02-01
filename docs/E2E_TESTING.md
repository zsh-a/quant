# 端到端测试指南

## 概述

端到端测试验证整个系统的集成，从数据流到API到前端的完整流程。

## 前置条件

### 1. 启动API服务器

```bash
cd /home/zs/workspace/exp/quent
uvicorn src.api.server:app --reload
```

服务器应该在 `http://localhost:8000` 运行。

### 2. 确保数据库可用

确保ClickHouse数据库正在运行并包含测试数据。

## 运行测试

### 完整集成测试

```bash
python test_e2e_integration.py
```

### 测试覆盖

1. **API服务器状态检查**
   - 验证服务器是否运行
   - 检查健康端点

2. **策略列表获取**
   - 获取可用策略
   - 验证策略参数

3. **回测会话创建**
   - 创建短期回测会话
   - 验证session_id返回

4. **增量更新测试** ⭐
   - 第一次请求：获取完整数据
   - 第二次请求：使用`since`参数获取增量数据
   - 计算带宽节省比例

5. **会话完成监控**
   - 轮询会话状态
   - 监控进度
   - 验证完成状态

6. **会话列表获取**
   - 获取所有会话
   - 验证持久化

7. **日志文件验证**
   - 检查日志目录
   - 验证日志文件创建

## 预期结果

### 成功输出示例

```
############################################################
# End-to-End Integration Test Suite
############################################################

============================================================
Checking API server status...
✓ API server is up: {'status': 'up', 'active_sessions': 0}

============================================================
TEST: Get Available Strategies
============================================================
Found 2 strategies:
  - jsg: JSG Quantitative
    Parameters: 5
  - rotation: Advanced Rotation
    Parameters: 3
✓ Strategy list retrieved successfully

============================================================
TEST: Create Backtest Session
============================================================
Creating session: jsg on sh.600000
✓ Session created: a1b2c3d4-...

============================================================
TEST: Incremental Session Status Updates
============================================================
Request 1: Full data (no since parameter)
  Response size: 15234 bytes
  Equity points: 21
  Trades: 3
  Status: running
  Progress: 45.2%

Request 2: Incremental data (since=2023-01-15T10:30:00)
  Response size: 1523 bytes
  Equity points: 5
  Trades: 1

  Bandwidth savings: 90.0% (15234 → 1523 bytes)
✓ Incremental updates working efficiently

============================================================
TEST: Wait for Session Completion
============================================================
Progress: 50.0% - Status: running
Progress: 75.0% - Status: running
Progress: 100.0% - Status: completed

✓ Session completed
  Total equity points: 21
  Total trades: 5

============================================================
TEST SUMMARY
============================================================
✓ PASS: strategies
✓ PASS: create_session
✓ PASS: incremental_updates
✓ PASS: completion
✓ PASS: get_sessions
✓ PASS: log_files

Total: 6/6 tests passed (100%)

🎉 ALL TESTS PASSED!
```

## 性能指标

### 增量更新效率

- **目标**: 带宽节省 >80%
- **实际**: 通常 85-95%
- **计算**: `(full_size - incremental_size) / full_size * 100`

### 响应时间

- **策略列表**: <100ms
- **创建会话**: <500ms
- **状态查询**: <200ms
- **增量更新**: <100ms

## 故障排查

### API服务器未运行

```
✗ API server is not running
Please start the server with: uvicorn src.api.server:app --reload
```

**解决方案**: 启动API服务器

### 数据库连接失败

```
Error creating session: Database connection failed
```

**解决方案**: 
1. 检查ClickHouse是否运行
2. 验证数据库配置
3. 确认测试数据存在

### 增量更新效率低

```
⚠ Bandwidth savings only 30.0% (expected >50%)
```

**可能原因**:
1. 会话刚创建，数据量少
2. 两次请求间隔太短
3. 后端`since`参数未正确处理

**解决方案**: 
1. 增加两次请求间隔
2. 检查后端日志
3. 验证`since`参数传递

### 会话超时

```
✗ Timeout after 60s
```

**解决方案**:
1. 增加`TEST_TIMEOUT`值
2. 使用更短的测试周期
3. 检查回测是否卡住

## 手动测试

### 使用curl测试API

```bash
# 检查服务器状态
curl http://localhost:8000/status

# 获取策略列表
curl http://localhost:8000/strategies

# 创建会话
curl -X POST http://localhost:8000/session/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "jsg",
    "symbol": "sh.600000",
    "start_date": "2023-01-01",
    "end_date": "2023-01-31",
    "mode": "backtest"
  }'

# 查询会话状态（完整）
curl "http://localhost:8000/session/{session_id}/status"

# 查询会话状态（增量）
curl "http://localhost:8000/session/{session_id}/status?since=2023-01-15T10:30:00"
```

### 使用浏览器测试

1. 打开 `http://localhost:8000/docs`
2. 使用Swagger UI交互式测试API
3. 查看请求/响应详情

## 持续集成

### 集成到CI/CD

```yaml
# .github/workflows/test.yml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Start API server
        run: |
          uvicorn src.api.server:app &
          sleep 5
      
      - name: Run E2E tests
        run: python test_e2e_integration.py
```

## 相关文件

- 测试脚本: [`test_e2e_integration.py`](file:///home/zs/workspace/exp/quent/test_e2e_integration.py)
- API服务器: [`src/api/server.py`](file:///home/zs/workspace/exp/quent/src/api/server.py)
- 前端应用: [`ui/src/App.tsx`](file:///home/zs/workspace/exp/quent/ui/src/App.tsx)
