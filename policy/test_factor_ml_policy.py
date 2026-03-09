#!/usr/bin/env python3
"""
测试 factor_ml_policy 的功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# 添加项目根目录到Python路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.market_data.db import DB
from policy.factor_ml_policy import Agent


def test_jointdata_connection():
    """测试 jointdata 连接"""
    print("🔗 测试 jointdata 连接...")
    
    try:
        from jointdata.wide_table_manager import WideTableManager
        
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
        
        # 获取因子列表
        factors = manager.get_available_factors()
        print(f"   可用因子数量: {len(factors)}")
        if factors:
            print(f"   前5个因子: {factors[:5]}")
        
        return True, manager
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False, None


def test_lgb_model_loading():
    """测试 LGB 模型加载"""
    print("\n🤖 测试 LGB 模型加载...")
    
    model_path = "./simple_models"
    model_name = "lgb_model_20241201.txt"
    model_file = os.path.join(model_path, model_name)
    
    if os.path.exists(model_file):
        print(f"✅ 模型文件存在: {model_file}")
        return True
    else:
        print(f"❌ 模型文件不存在: {model_file}")
        print("请确保模型文件存在")
        return False


def test_agent_initialization():
    """测试代理初始化"""
    print("\n🚀 测试代理初始化...")
    
    try:
        # 初始化数据库客户端
        db_client = DB()
        
        # 测试配置
        test_config = {
            "db_client": db_client,
            "model_path": "./simple_models",
            "model_name": "lgb_model_20241201.txt",
            "use_simple_model": True,
            "use_jointdata": True,
            "jointdata_config": {
                'host': 'localhost',
                'port': 8123,
                'user': 'default',
                'password': '',
                'database': 'factor_db',
                'table_name': 'factor_data'
            },
            "prediction_threshold": 0.6,
            "sell_threshold": 0.4,
            "max_stocks": 10,
        }
        
        # 创建代理（不传入 market_env，仅测试初始化）
        agent = Agent(None, **test_config)
        
        print("✅ 代理初始化成功")
        print(f"   模型加载状态: {agent.model_loaded}")
        print(f"   jointdata 状态: {agent.use_jointdata}")
        print(f"   最大持股数量: {agent.max_stocks}")
        
        return True
        
    except Exception as e:
        print(f"❌ 代理初始化失败: {e}")
        return False


def test_factor_calculation():
    """测试因子计算"""
    print("\n📊 测试因子计算...")
    
    try:
        # 初始化数据库客户端
        db_client = DB()
        
        # 测试配置
        test_config = {
            "db_client": db_client,
            "model_path": "./simple_models",
            "model_name": "lgb_model_20241201.txt",
            "use_simple_model": True,
            "use_jointdata": True,
            "jointdata_config": {
                'host': 'localhost',
                'port': 8123,
                'user': 'default',
                'password': '',
                'database': 'factor_db',
                'table_name': 'factor_data'
            },
            "prediction_threshold": 0.6,
            "sell_threshold": 0.4,
            "max_stocks": 10,
        }
        
        # 创建代理
        agent = Agent(None, **test_config)
        
        # 设置当前日期
        agent.current_date = pd.Timestamp('2023-12-01')
        
        # 测试股票池
        test_stocks = ['000001.SZ', '000002.SZ', '000858.SZ']
        
        # 测试因子获取
        factors = agent.get_stock_factors(test_stocks, agent.current_date)
        
        if factors is not None and not factors.empty:
            print("✅ 因子计算成功")
            print(f"   股票数量: {len(factors)}")
            print(f"   因子数量: {len(factors.columns)}")
            print(f"   因子列: {list(factors.columns[:5])}...")
        else:
            print("❌ 因子计算失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 因子计算测试失败: {e}")
        return False


def test_prediction():
    """测试预测功能"""
    print("\n🔮 测试预测功能...")
    
    try:
        # 初始化数据库客户端
        db_client = DB()
        
        # 测试配置
        test_config = {
            "db_client": db_client,
            "model_path": "./simple_models",
            "model_name": "lgb_model_20241201.txt",
            "use_simple_model": True,
            "use_jointdata": True,
            "jointdata_config": {
                'host': 'localhost',
                'port': 8123,
                'user': 'default',
                'password': '',
                'database': 'factor_db',
                'table_name': 'factor_data'
            },
            "prediction_threshold": 0.6,
            "sell_threshold": 0.4,
            "max_stocks": 10,
        }
        
        # 创建代理
        agent = Agent(None, **test_config)
        
        # 设置当前日期
        agent.current_date = pd.Timestamp('2023-12-01')
        
        # 测试股票池
        test_stocks = ['000001.SZ', '000002.SZ', '000858.SZ']
        
        # 测试预测
        predictions = agent.predict_stock_probabilities(test_stocks, agent.current_date)
        
        if predictions:
            print("✅ 预测功能正常")
            print(f"   预测股票数量: {len(predictions)}")
            for code, prob in predictions.items():
                print(f"   {code}: {prob:.4f}")
        else:
            print("❌ 预测功能失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 预测测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 开始测试 factor_ml_policy 功能")
    print("=" * 50)
    
    # 测试结果
    test_results = {}
    
    # 1. 测试 jointdata 连接
    test_results['jointdata'] = test_jointdata_connection()
    
    # 2. 测试 LGB 模型加载
    test_results['lgb_model'] = test_lgb_model_loading()
    
    # 3. 测试代理初始化
    test_results['agent_init'] = test_agent_initialization()
    
    # 4. 测试因子计算
    if test_results['agent_init']:
        test_results['factor_calc'] = test_factor_calculation()
    
    # 5. 测试预测功能
    if test_results['agent_init']:
        test_results['prediction'] = test_prediction()
    
    # 输出测试结果
    print("\n📋 测试结果总结")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15}: {status}")
    
    # 总体评估
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！factor_ml_policy 功能正常")
    else:
        print("⚠️  部分测试失败，请检查相关配置")


if __name__ == "__main__":
    main() 
