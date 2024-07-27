# 指数市场分析器

这是一个用于分析指数历史数据牛熊阶段的Python工具，可以从数据库中读取指数数据，自动识别牛市和熊市阶段，并生成详细的分析报告和可视化图表。

## 功能特性

- **自动识别牛熊阶段**: 基于累计收益率的极值点自动识别市场阶段
- **技术指标计算**: 计算移动平均线、RSI、MACD、波动率等技术指标
- **可视化分析**: 生成价格走势、技术指标、阶段统计等多种图表
- **数据导出**: 将分析结果保存为CSV和JSON格式
- **多指数支持**: 支持分析上证指数、深证成指、沪深300等多种指数

## 安装依赖

确保已安装以下Python包：

```bash
pip install numpy pandas matplotlib seaborn
```

或者使用项目的虚拟环境：

```bash
source .venv/bin/activate
```

## 使用方法

### 基本使用

```python
from index_market_analysis import IndexMarketAnalyzer
from db import DB

# 创建数据库连接
db = DB()

# 创建分析器
analyzer = IndexMarketAnalyzer(db)

# 加载指数数据
analyzer.load_index_data(
    index_code="sh.000001",  # 上证指数
    start_date="20100101",   # 开始日期
    end_date="20241231"      # 结束日期
)

# 识别市场阶段
analyzer.identify_market_phases(
    min_phase_days=10,           # 最小阶段天数
    min_return_threshold=0.05,   # 牛市最小收益率阈值
    min_drawdown_threshold=-0.05 # 熊市最小回撤阈值
)

# 打印分析摘要
analyzer.print_phase_summary()

# 创建可视化图表
analyzer.create_visualizations(save_plots=True)

# 保存分析结果
analyzer.save_analysis_results()
```

### 运行示例

运行基本示例：

```bash
python index_market_analysis.py
```

运行完整示例（分析多个指数）：

```bash
python example_index_analysis.py
```

## 参数说明

### 市场阶段识别参数

- `min_phase_days`: 最小阶段天数，默认60天
- `min_return_threshold`: 牛市最小收益率阈值，默认0.2（20%）
- `min_drawdown_threshold`: 熊市最小回撤阈值，默认-0.15（-15%）

### 支持的指数代码

- `sh.000001`: 上证指数
- `sz.399001`: 深证成指
- `sh.000300`: 沪深300指数
- `sh.000905`: 中证500指数
- `sz.399006`: 创业板指

## 输出文件

### 图表文件

- `price_with_phases.png`: 价格走势与牛熊阶段标注 (英文标签)
- `technical_indicators.png`: 技术指标分析 (英文标签)
- `phase_statistics.png`: 阶段统计对比 (英文标签)

### 数据文件

- `market_phases.csv`: 市场阶段详细数据
- `index_data.csv`: 原始指数数据（包含技术指标）
- `analysis_statistics.json`: 分析统计信息

## 分析结果解读

### 市场阶段信息

每个识别出的市场阶段包含以下信息：

- `start_date`: 阶段开始日期
- `end_date`: 阶段结束日期
- `phase_type`: 阶段类型（bull/bear）
- `duration_days`: 持续时间（天）
- `total_return`: 总收益率
- `max_drawdown`: 最大回撤
- `volatility`: 波动率

### 统计指标

- **牛市统计**: 平均持续时间、平均收益率、平均波动率
- **熊市统计**: 平均持续时间、平均最大回撤、平均波动率
- **整体统计**: 总阶段数、牛熊比例、平均阶段时长

## 可视化图表说明

### 1. 价格走势与牛熊阶段

- 上图：显示指数价格走势，用不同颜色标注牛市和熊市阶段 (英文标签)
- 下图：显示累计收益率走势 (英文标签)

### 2. 技术指标分析

- 移动平均线：MA20、MA60等 (英文标签)
- 回撤走势：显示历史回撤情况 (英文标签)

### 3. 阶段统计对比

- 阶段数量统计：牛市和熊市阶段数量对比 (英文标签)
- 持续时间对比：牛市和熊市平均持续时间对比 (英文标签)

## 自定义分析

### 调整识别参数

```python
# 更严格的识别条件
analyzer.identify_market_phases(
    min_phase_days=60,           # 更长的最小阶段
    min_return_threshold=0.2,    # 更高的收益率要求
    min_drawdown_threshold=-0.15 # 更大的回撤要求
)

# 更宽松的识别条件
analyzer.identify_market_phases(
    min_phase_days=5,            # 更短的最小阶段
    min_return_threshold=0.03,   # 更低的收益率要求
    min_drawdown_threshold=-0.03 # 更小的回撤要求
)
```

### 分析不同时间段

```python
# 分析特定时间段
analyzer.load_index_data(
    index_code="sh.000001",
    start_date="20200101",  # 2020年开始
    end_date="20241231"     # 2024年结束
)
```

## 注意事项

1. **数据质量**: 确保数据库中的指数数据完整且准确
2. **参数调整**: 根据分析需求调整识别参数，避免过度拟合
3. **结果解释**: 分析结果仅供参考，不构成投资建议
4. **性能考虑**: 分析大量数据时可能需要较长时间

## 故障排除

### 常见问题

1. **没有识别到市场阶段**
   - 降低阈值参数
   - 检查数据质量
   - 增加分析时间范围

2. **图表显示异常**
   - 检查matplotlib配置
   - 确保中文字体支持

3. **数据库连接失败**
   - 检查数据库配置
   - 确认网络连接

## 扩展功能

### 添加新的技术指标

可以在 `_calculate_technical_indicators` 方法中添加新的技术指标计算。

### 自定义可视化

可以修改 `_plot_*` 方法来创建自定义的图表。

### 集成其他数据源

可以修改 `load_index_data` 方法来支持其他数据源。

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交Issue或联系开发者。 