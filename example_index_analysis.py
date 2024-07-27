#!/usr/bin/env python3
"""
指数市场分析器使用示例

这个文件展示了如何使用 IndexMarketAnalyzer 来分析不同指数的牛熊阶段
"""

import os
import sys
from datetime import datetime

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from index_market_analysis import IndexMarketAnalyzer
from db import DB


def analyze_shanghai_index():
    """分析上证指数"""
    print("=" * 60)
    print("分析上证指数 (sh.000001)")
    print("=" * 60)
    
    db = DB()
    analyzer = IndexMarketAnalyzer(db)
    
    # 加载数据
    analyzer.load_index_data(
        index_code="sh.000001",
        start_date="20100101",
        end_date="20241231"
    )
    
    # 识别市场阶段
    analyzer.identify_market_phases(
        min_phase_days=10,           # 最小阶段10天
        min_return_threshold=0.05,   # 牛市最小收益率5%
        min_drawdown_threshold=-0.05 # 熊市最小回撤5%
    )
    
    # 打印分析摘要
    analyzer.print_phase_summary()
    
    # 创建可视化
    analyzer.create_visualizations(
        save_plots=True,
        plot_dir="./shanghai_index_analysis"
    )
    
    # 保存结果
    analyzer.save_analysis_results("./shanghai_index_results")
    
    return analyzer


def analyze_shenzhen_index():
    """分析深证成指"""
    print("\n" + "=" * 60)
    print("分析深证成指 (sz.399001)")
    print("=" * 60)
    
    db = DB()
    analyzer = IndexMarketAnalyzer(db)
    
    # 加载数据
    analyzer.load_index_data(
        index_code="sz.399001",
        start_date="20100101",
        end_date="20241231"
    )
    
    # 识别市场阶段
    analyzer.identify_market_phases(
        min_phase_days=10,
        min_return_threshold=0.05,
        min_drawdown_threshold=-0.05
    )
    
    # 打印分析摘要
    analyzer.print_phase_summary()
    
    # 创建可视化
    analyzer.create_visualizations(
        save_plots=True,
        plot_dir="./shenzhen_index_analysis"
    )
    
    # 保存结果
    analyzer.save_analysis_results("./shenzhen_index_results")
    
    return analyzer


def analyze_hs300_index():
    """分析沪深300指数"""
    print("\n" + "=" * 60)
    print("分析沪深300指数 (sh.000300)")
    print("=" * 60)
    
    db = DB()
    analyzer = IndexMarketAnalyzer(db)
    
    # 加载数据
    analyzer.load_index_data(
        index_code="sh.000300",
        start_date="20100101",
        end_date="20241231"
    )
    
    # 识别市场阶段
    analyzer.identify_market_phases(
        min_phase_days=10,
        min_return_threshold=0.05,
        min_drawdown_threshold=-0.05
    )
    
    # 打印分析摘要
    analyzer.print_phase_summary()
    
    # 创建可视化
    analyzer.create_visualizations(
        save_plots=True,
        plot_dir="./hs300_index_analysis"
    )
    
    # 保存结果
    analyzer.save_analysis_results("./hs300_index_results")
    
    return analyzer


def compare_indices():
    """比较不同指数的表现"""
    print("\n" + "=" * 60)
    print("比较不同指数的牛熊阶段表现")
    print("=" * 60)
    
    indices = [
        ("sh.000001", "上证指数"),
        ("sz.399001", "深证成指"),
        ("sh.000300", "沪深300")
    ]
    
    results = {}
    
    for index_code, index_name in indices:
        print(f"\n分析 {index_name} ({index_code})...")
        
        try:
            db = DB()
            analyzer = IndexMarketAnalyzer(db)
            
            # 加载数据
            analyzer.load_index_data(
                index_code=index_code,
                start_date="20100101",
                end_date="20241231"
            )
            
            # 识别市场阶段
            analyzer.identify_market_phases(
                min_phase_days=10,
                min_return_threshold=0.05,
                min_drawdown_threshold=-0.05
            )
            
            # 统计结果
            bull_phases = [p for p in analyzer.market_phases if p.phase_type == 'bull']
            bear_phases = [p for p in analyzer.market_phases if p.phase_type == 'bear']
            
            results[index_name] = {
                'total_phases': len(analyzer.market_phases),
                'bull_phases': len(bull_phases),
                'bear_phases': len(bear_phases),
                'avg_bull_return': sum(p.total_return for p in bull_phases) / len(bull_phases) if bull_phases else 0,
                'avg_bear_drawdown': sum(p.max_drawdown for p in bear_phases) / len(bear_phases) if bear_phases else 0,
                'avg_bull_duration': sum(p.duration_days for p in bull_phases) / len(bull_phases) if bull_phases else 0,
                'avg_bear_duration': sum(p.duration_days for p in bear_phases) / len(bear_phases) if bear_phases else 0
            }
            
        except Exception as e:
            print(f"分析 {index_name} 时出错: {e}")
            results[index_name] = None
    
    # 打印比较结果
    print("\n" + "=" * 80)
    print("指数比较结果")
    print("=" * 80)
    print(f"{'指数名称':<12} {'总阶段':<6} {'牛市':<6} {'熊市':<6} {'平均牛市收益':<12} {'平均熊市回撤':<12} {'平均牛市天数':<12} {'平均熊市天数':<12}")
    print("-" * 80)
    
    for index_name, result in results.items():
        if result is not None:
            print(f"{index_name:<12} {result['total_phases']:<6} {result['bull_phases']:<6} {result['bear_phases']:<6} "
                  f"{result['avg_bull_return']:.2%:<10} {result['avg_bear_drawdown']:.2%:<10} "
                  f"{result['avg_bull_duration']:.0f}天{'':<8} {result['avg_bear_duration']:.0f}天{'':<8}")
        else:
            print(f"{index_name:<12} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")


def analyze_custom_period():
    """分析自定义时间段"""
    print("\n" + "=" * 60)
    print("分析自定义时间段 (2020-2024)")
    print("=" * 60)
    
    db = DB()
    analyzer = IndexMarketAnalyzer(db)
    
    # 加载数据
    analyzer.load_index_data(
        index_code="sh.000001",
        start_date="20200101",
        end_date="20241231"
    )
    
    # 识别市场阶段
    analyzer.identify_market_phases(
        min_phase_days=5,            # 较短的最小阶段
        min_return_threshold=0.03,   # 较低的收益率阈值
        min_drawdown_threshold=-0.03 # 较低的回撤阈值
    )
    
    # 打印分析摘要
    analyzer.print_phase_summary()
    
    # 创建可视化
    analyzer.create_visualizations(
        save_plots=True,
        plot_dir="./custom_period_analysis"
    )
    
    return analyzer


def main():
    """主函数"""
    print("指数市场分析器使用示例")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 分析上证指数
        shanghai_analyzer = analyze_shanghai_index()
        
        # 2. 分析深证成指
        shenzhen_analyzer = analyze_shenzhen_index()
        
        # 3. 分析沪深300指数
        hs300_analyzer = analyze_hs300_index()
        
        # 4. 比较不同指数
        compare_indices()
        
        # 5. 分析自定义时间段
        custom_analyzer = analyze_custom_period()
        
        print("\n" + "=" * 60)
        print("所有分析完成!")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print("\nGenerated files:")
        print("- Shanghai Index Analysis: ./shanghai_index_analysis/")
        print("- Shenzhen Index Analysis: ./shenzhen_index_analysis/")
        print("- CSI 300 Analysis: ./hs300_index_analysis/")
        print("- Custom Period Analysis: ./custom_period_analysis/")
        print("- Result Data: ./*_index_results/")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 