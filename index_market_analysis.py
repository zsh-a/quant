import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List
from dataclasses import dataclass
import json

# 设置英文样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 将外层目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.market_data.db import DB

warnings.filterwarnings('ignore')


@dataclass
class MarketPhase:
    """市场阶段数据类"""
    start_date: str
    end_date: str
    phase_type: str  # 'bull' 或 'bear'
    duration_days: int
    total_return: float
    max_drawdown: float
    volatility: float


class IndexMarketAnalyzer:
    """指数市场分析器"""
    
    def __init__(self, db: DB):
        self.db = db
        self.index_data = None
        self.market_phases = []
        
    def load_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库加载指数历史数据"""
        print(f"正在加载指数 {index_code} 的历史数据...")
        
        fields = ["open", "close", "high", "low", "volume", "amount"]
        
        data = self.db.get_price(
            stocks=[index_code],
            end_date=end_date,
            fields=fields,
            count=9999999,
            start_date=start_date,
        )
        
        if data.empty:
            raise ValueError(f"未找到指数 {index_code} 的数据")
        
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # 计算技术指标
        data['daily_return'] = data['close'].pct_change()
        data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1
        data['ma20'] = data['close'].rolling(window=20).mean()
        data['ma60'] = data['close'].rolling(window=60).mean()
        data['volatility'] = data['daily_return'].rolling(window=20).std() * np.sqrt(252)
        data['rolling_max'] = data['close'].rolling(window=252, min_periods=1).max()
        data['drawdown'] = (data['close'] - data['rolling_max']) / data['rolling_max']
        
        self.index_data = data
        print(f"成功加载 {len(data)} 条指数数据")
        return data
    
    def identify_market_phases(self, min_phase_days=60, min_return_threshold=0.2, min_drawdown_threshold=-0.15):
        """识别市场牛熊阶段"""
        if self.index_data is None:
            raise ValueError("请先加载指数数据")
        
        print("正在识别市场牛熊阶段...")
        
        data = self.index_data.copy()
        phases = []
        
        cumulative_returns = data['cumulative_return'].values
        dates = data['date'].values
        prices = data['close'].values
        
        # 寻找局部极值点
        peaks = []
        troughs = []
        
        for i in range(1, len(cumulative_returns) - 1):
            if (cumulative_returns[i] > cumulative_returns[i-1] and 
                cumulative_returns[i] > cumulative_returns[i+1]):
                peaks.append(i)
            
            if (cumulative_returns[i] < cumulative_returns[i-1] and 
                cumulative_returns[i] < cumulative_returns[i+1]):
                troughs.append(i)
        
        print(f"找到 {len(peaks)} 个峰值点和 {len(troughs)} 个谷值点")
        if len(peaks) < 2 or len(troughs) < 2:
            print("警告: 极值点不足")
            return phases
        
        all_extrema = sorted([(i, 'peak') for i in peaks] + [(i, 'trough') for i in troughs])
        
        print(f"开始分析 {len(all_extrema)} 个极值点...")
        for i in range(len(all_extrema) - 1):
            current_idx, current_type = all_extrema[i]
            next_idx, next_type = all_extrema[i + 1]
            
            phase_days = int((dates[next_idx] - dates[current_idx]) / np.timedelta64(1, 'D'))
            if phase_days < min_phase_days:
                continue
            
            phase_return = (prices[next_idx] - prices[current_idx]) / prices[current_idx]
            
            phase_prices = prices[current_idx:next_idx+1]
            phase_rolling_max = np.maximum.accumulate(phase_prices)
            phase_drawdown = (phase_prices - phase_rolling_max) / phase_rolling_max
            max_drawdown = np.min(phase_drawdown)
            
            phase_returns = data['daily_return'].iloc[current_idx:next_idx+1].dropna()
            volatility = phase_returns.std() * np.sqrt(252) if len(phase_returns) > 0 else 0
            
            # 调试信息（可选）
            # if i < 10:  # 只打印前10个阶段的信息
            #     print(f"阶段 {i}: {current_type}->{next_type}, 天数:{phase_days}, 收益率:{phase_return:.3f}, 回撤:{max_drawdown:.3f}")
            
            if current_type == 'trough' and next_type == 'peak' and phase_return >= min_return_threshold:
                phase_type = 'bull'
            elif current_type == 'peak' and next_type == 'trough' and max_drawdown <= min_drawdown_threshold:
                phase_type = 'bear'
            else:
                continue
            
            phase = MarketPhase(
                start_date=str(dates[current_idx])[:10],
                end_date=str(dates[next_idx])[:10],
                phase_type=phase_type,
                duration_days=phase_days,
                total_return=phase_return,
                max_drawdown=max_drawdown,
                volatility=volatility
            )
            
            phases.append(phase)
        
        self.market_phases = phases
        
        bull_phases = [p for p in phases if p.phase_type == 'bull']
        bear_phases = [p for p in phases if p.phase_type == 'bear']
        print(f"识别出 {len(phases)} 个市场阶段")
        print(f"牛市阶段: {len(bull_phases)} 个")
        print(f"熊市阶段: {len(bear_phases)} 个")
        
        return phases
    
    def print_phase_summary(self):
        """打印市场阶段摘要"""
        if not self.market_phases:
            print("没有识别到市场阶段")
            return
        
        print("\n" + "="*60)
        print("市场阶段分析摘要")
        print("="*60)
        
        bull_phases = [p for p in self.market_phases if p.phase_type == 'bull']
        bear_phases = [p for p in self.market_phases if p.phase_type == 'bear']
        
        print(f"总阶段数: {len(self.market_phases)}")
        print(f"牛市阶段: {len(bull_phases)}")
        print(f"熊市阶段: {len(bear_phases)}")
        
        if bull_phases:
            avg_bull_duration = np.mean([p.duration_days for p in bull_phases])
            avg_bull_return = np.mean([p.total_return for p in bull_phases])
            print(f"\n牛市统计:")
            print(f"  平均持续时间: {avg_bull_duration:.0f} 天")
            print(f"  平均收益率: {avg_bull_return:.2%}")
        
        if bear_phases:
            avg_bear_duration = np.mean([p.duration_days for p in bear_phases])
            avg_bear_drawdown = np.mean([p.max_drawdown for p in bear_phases])
            print(f"\n熊市统计:")
            print(f"  平均持续时间: {avg_bear_duration:.0f} 天")
            print(f"  平均最大回撤: {avg_bear_drawdown:.2%}")
        
        print(f"\n详细阶段信息:")
        print("-" * 60)
        print(f"{'类型':<6} {'开始日期':<12} {'结束日期':<12} {'持续天数':<8} {'收益率/回撤':<12}")
        print("-" * 60)
        
        for phase in self.market_phases:
            if phase.phase_type == 'bull':
                return_str = f"{phase.total_return:.2%}"
            else:
                return_str = f"{phase.max_drawdown:.2%}"
            
            print(f"{'牛市' if phase.phase_type == 'bull' else '熊市':<6} "
                  f"{phase.start_date:<12} {phase.end_date:<12} {phase.duration_days:<8} "
                  f"{return_str:<12}")
    
    def create_visualizations(self, save_plots=True, plot_dir="./market_analysis_plots"):
        """创建可视化图表"""
        if self.index_data is None:
            raise ValueError("请先加载指数数据")
        
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
        
        print("正在创建可视化图表...")
        
        # 价格走势和牛熊阶段
        self._plot_price_with_phases(save_plots, plot_dir)
        
        # 技术指标
        self._plot_technical_indicators(save_plots, plot_dir)
        
        # 阶段统计
        self._plot_phase_statistics(save_plots, plot_dir)
        
        print("可视化图表创建完成!")
    
    def _plot_price_with_phases(self, save_plots, plot_dir):
        """绘制价格走势和牛熊阶段标注"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        data = self.index_data
        
        # 价格走势
        ax1.plot(data['date'], data['close'], label='Close Price', linewidth=1, color='black')
        
        colors = {'bull': 'red', 'bear': 'green'}
        for phase in self.market_phases:
            start_date = pd.to_datetime(phase.start_date)
            end_date = pd.to_datetime(phase.end_date)
            phase_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            
            if not phase_data.empty:
                color = colors[phase.phase_type]
                label = 'Bull Market' if phase.phase_type == 'bull' else 'Bear Market'
                ax1.plot(phase_data['date'], phase_data['close'], 
                        color=color, linewidth=2, label=f'{label} ({phase.duration_days}d)')
        
        ax1.set_title('Index Price Trend with Bull/Bear Phases', fontsize=16)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累计收益率
        ax2.plot(data['date'], data['cumulative_return'] * 100, 
                label='Cumulative Return', linewidth=1, color='blue')
        
        for phase in self.market_phases:
            start_date = pd.to_datetime(phase.start_date)
            end_date = pd.to_datetime(phase.end_date)
            phase_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            
            if not phase_data.empty:
                color = colors[phase.phase_type]
                ax2.plot(phase_data['date'], phase_data['cumulative_return'] * 100, 
                        color=color, linewidth=2)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Return Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/price_with_phases.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_technical_indicators(self, save_plots, plot_dir):
        """绘制技术指标"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        data = self.index_data
        
        # 移动平均线
        ax1 = axes[0]
        ax1.plot(data['date'], data['close'], label='Close Price', linewidth=1, color='black')
        ax1.plot(data['date'], data['ma20'], label='MA20', linewidth=1, color='red')
        ax1.plot(data['date'], data['ma60'], label='MA60', linewidth=1, color='blue')
        ax1.set_title('Moving Average Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 回撤
        ax2 = axes[1]
        ax2.fill_between(data['date'], data['drawdown'] * 100, 0, 
                        where=(data['drawdown'] <= 0), color='red', alpha=0.3)
        ax2.plot(data['date'], data['drawdown'] * 100, linewidth=1, color='red')
        ax2.set_title('Drawdown Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/technical_indicators.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_phase_statistics(self, save_plots, plot_dir):
        """绘制阶段统计"""
        if not self.market_phases:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        bull_phases = [p for p in self.market_phases if p.phase_type == 'bull']
        bear_phases = [p for p in self.market_phases if p.phase_type == 'bear']
        
        # 阶段数量
        ax1 = axes[0]
        phase_counts = [len(bull_phases), len(bear_phases)]
        ax1.bar(['Bull Market', 'Bear Market'], phase_counts, color=['red', 'green'], alpha=0.7)
        ax1.set_title('Phase Count Statistics')
        ax1.set_ylabel('Number of Phases')
        
        for i, count in enumerate(phase_counts):
            ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # 平均持续时间
        ax2 = axes[1]
        if bull_phases and bear_phases:
            bull_durations = [p.duration_days for p in bull_phases]
            bear_durations = [p.duration_days for p in bear_phases]
            
            ax2.boxplot([bull_durations, bear_durations], labels=['Bull Market', 'Bear Market'])
            ax2.set_title('Phase Duration Comparison')
            ax2.set_ylabel('Days')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/phase_statistics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_results(self, output_dir="./market_analysis_results"):
        """保存分析结果"""
        if not self.market_phases:
            print("没有分析结果可保存")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存阶段数据
        phase_data = []
        for phase in self.market_phases:
            phase_data.append({
                'start_date': phase.start_date,
                'end_date': phase.end_date,
                'phase_type': phase.phase_type,
                'duration_days': phase.duration_days,
                'total_return': phase.total_return,
                'max_drawdown': phase.max_drawdown,
                'volatility': phase.volatility
            })
        
        phase_df = pd.DataFrame(phase_data)
        phase_file = os.path.join(output_dir, "market_phases.csv")
        phase_df.to_csv(phase_file, index=False, encoding='utf-8-sig')
        print(f"阶段数据已保存到: {phase_file}")
        
        # 保存指数数据
        if self.index_data is not None:
            data_file = os.path.join(output_dir, "index_data.csv")
            self.index_data.to_csv(data_file, index=False, encoding='utf-8-sig')
            print(f"指数数据已保存到: {data_file}")


def main():
    """主函数示例"""
    # 创建数据库连接
    db = DB()
    
    # 创建分析器
    analyzer = IndexMarketAnalyzer(db)
    
    # 分析上证指数
    index_code = "sz.399101"  # 上证指数
    start_date = "20100101"   # 2010年开始
    end_date = "20241231"     # 2024年结束
    
    try:
        # 加载数据
        analyzer.load_index_data(index_code, start_date, end_date)
        
        # 识别市场阶段
        analyzer.identify_market_phases(
            min_phase_days=10,           # 最小阶段10天
            min_return_threshold=0.05,   # 牛市最小收益率5%
            min_drawdown_threshold=-0.05 # 熊市最小回撤5%
        )
        
        # 打印分析摘要
        analyzer.print_phase_summary()
        
        # 创建可视化
        analyzer.create_visualizations(save_plots=True)
        
        # 保存分析结果
        analyzer.save_analysis_results()
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
