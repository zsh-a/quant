#!/usr/bin/env python3
"""
使用 factor_ml_policy 进行回测的示例
支持 jointdata 特征和 LGB 模型
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

# 添加项目根目录到Python路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.market_data.db import DB
from market_env import MultiMarketEnv
from policy.factor_ml_policy import Agent, OrderPolicy


def run_backtest_with_jointdata():
    """使用 jointdata 特征进行回测"""
    print("🚀 开始使用 jointdata 特征进行回测...")
    
    # 初始化数据库客户端
    db_client = DB()
    
    # 回测参数配置
    backtest_config = {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 1000000,  # 100万初始资金
        "commission_rate": 0.0003,   # 手续费率
        "slippage": 0.002,          # 滑点
    }
    
    # 策略参数配置
    strategy_config = {
        "db_client": db_client,
        "model_path": "./simple_models",
        "model_name": "lgb_model_20241201.txt",  # LGB模型文件
        "use_simple_model": True,
        "use_jointdata": True,  # 启用 jointdata 支持
        "jointdata_config": {
            'host': 'localhost',
            'port': 8123,
            'user': 'default',
            'password': '',
            'database': 'factor_db',
            'table_name': 'factor_data'
        },
        "prediction_threshold": 0.6,  # 买入阈值
        "sell_threshold": 0.4,        # 卖出阈值
        "max_stocks": 10,             # 最大持股数量
    }
    
    try:
        # 创建市场环境
        market_env = MultiMarketEnv(
            db_client=db_client,
            start_date=backtest_config["start_date"],
            end_date=backtest_config["end_date"],
            initial_capital=backtest_config["initial_capital"],
            commission_rate=backtest_config["commission_rate"],
            slippage=backtest_config["slippage"]
        )
        
        # 创建策略代理
        agent = Agent(market_env, **strategy_config)
        
        # 创建订单策略
        order_policy = OrderPolicy(market_env.account, **strategy_config)
        
        print(f"✅ 策略初始化完成")
        print(f"   回测期间: {backtest_config['start_date']} 到 {backtest_config['end_date']}")
        print(f"   初始资金: {backtest_config['initial_capital']:,}")
        print(f"   最大持股: {strategy_config['max_stocks']}")
        print(f"   买入阈值: {strategy_config['prediction_threshold']}")
        print(f"   卖出阈值: {strategy_config['sell_threshold']}")
        
        # 执行回测
        print("\n📈 开始执行回测...")
        
        # 这里需要根据实际的 market_env 接口来实现回测循环
        # 由于 market_env 的具体实现可能不同，这里提供一个框架
        
        # 示例回测循环（需要根据实际接口调整）
        """
        for date in market_env.get_trading_dates():
            # 更新当前日期
            agent.current_date = date
            order_policy.current_date = date
            
            # 获取市场数据
            market_data = market_env.get_market_data(date)
            
            # 执行策略决策
            agent.action_decider(market_data)
            
            # 执行订单
            market_env.execute_orders()
            
            # 记录结果
            market_env.record_daily_result()
            
            # 打印进度
            if date.day == 1:  # 每月第一天打印进度
                print(f"回测进度: {date.strftime('%Y-%m')}")
        """
        
        print("✅ 回测完成")
        
        # 分析回测结果
        analyze_backtest_results(market_env)
        
    except Exception as e:
        logger.error(f"回测执行失败: {str(e)}")
        raise


def run_backtest_with_alpha_factors():
    """使用 alpha 因子进行回测（备用方案）"""
    print("🚀 开始使用 alpha 因子进行回测...")
    
    # 初始化数据库客户端
    db_client = DB()
    
    # 策略参数配置（不使用 jointdata）
    strategy_config = {
        "db_client": db_client,
        "model_path": "./simple_models",
        "model_name": "simple_factor_model_20250608_183915",
        "use_simple_model": True,
        "use_jointdata": False,  # 不使用 jointdata
        "prediction_threshold": 0.6,
        "sell_threshold": 0.4,
        "max_stocks": 10,
    }
    
    try:
        # 创建市场环境
        market_env = MultiMarketEnv(
            db_client=db_client,
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=1000000,
            commission_rate=0.0003,
            slippage=0.002
        )
        
        # 创建策略代理
        agent = Agent(market_env, **strategy_config)
        
        print(f"✅ 策略初始化完成（使用 alpha 因子）")
        
        # 执行回测逻辑...
        
    except Exception as e:
        logger.error(f"回测执行失败: {str(e)}")
        raise


def analyze_backtest_results(market_env):
    """分析回测结果"""
    print("\n📊 回测结果分析")
    print("=" * 50)
    
    try:
        # 获取账户信息
        account = market_env.account
        
        # 计算收益率
        initial_value = account.initial_capital
        final_value = account.get_total_value()
        total_return = (final_value - initial_value) / initial_value * 100
        
        print(f"初始资金: {initial_value:,.2f}")
        print(f"最终资金: {final_value:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        
        # 获取持仓信息
        positions = account.get_positions()
        if positions:
            print(f"\n当前持仓:")
            for code, position in positions.items():
                print(f"  {code}: {position.quantity} 股")
        
        # 计算其他指标（如果有相关数据）
        # 夏普比率、最大回撤、胜率等
        
    except Exception as e:
        logger.error(f"结果分析失败: {str(e)}")


def test_jointdata_connection():
    """测试 jointdata 连接"""
    print("🔗 测试 jointdata 数据库连接...")
    
    try:
        from jointdata.wide_table_manager import WideTableManager
        
        # 创建连接
        manager = WideTableManager(
            host='localhost',
            port=8123,
            user='default',
            password='',
            database='factor_db',
            table_name='factor_data'
        )
        
        # 获取数据信息
        info = manager.get_data_info()
        print(f"✅ 连接成功")
        print(f"   数据范围: {info.get('start_date')} 到 {info.get('end_date')}")
        print(f"   股票数量: {info.get('stock_count')}")
        print(f"   总记录数: {info.get('total_records')}")
        
        # 获取可用因子
        factors = manager.get_available_factors()
        print(f"   可用因子数量: {len(factors)}")
        if factors:
            print(f"   前5个因子: {factors[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def main():
    """主函数"""
    print("🎯 因子ML策略回测示例")
    print("=" * 50)
    
    # 测试 jointdata 连接
    jointdata_available = test_jointdata_connection()
    
    if jointdata_available:
        print("\n选择回测模式:")
        print("1. 使用 jointdata 特征")
        print("2. 使用 alpha 因子")
        
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            run_backtest_with_jointdata()
        elif choice == "2":
            run_backtest_with_alpha_factors()
        else:
            print("无效选择，使用 jointdata 特征")
            run_backtest_with_jointdata()
    else:
        print("\n⚠️  jointdata 不可用，使用 alpha 因子进行回测")
        run_backtest_with_alpha_factors()


if __name__ == "__main__":
    main() 
