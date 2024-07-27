#!/usr/bin/env python3
"""
预计算因子数据脚本

这个脚本用于预计算所有日期的因子数据并缓存到磁盘中，
避免在模型训练和策略运行时重复计算因子。

使用方法:
python precompute_factors.py --start_date 20100101 --end_date 20241231 --batch_days 30

参数说明:
- start_date: 开始日期 (YYYYMMDD格式)
- end_date: 结束日期 (YYYYMMDD格式)  
- batch_days: 批处理天数，一次处理多少天的数据
- force_recompute: 是否强制重新计算（忽略已有缓存）
- stock_pool: 股票池类型 (csi500, csi300, all)
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha.factor_model import FactorModel
from db import DB


def main():
    parser = argparse.ArgumentParser(description='预计算因子数据')
    
    parser.add_argument('--start_date', type=str, default='20100101',
                        help='开始日期 (YYYYMMDD格式)')
    parser.add_argument('--end_date', type=str, default='20241231',
                        help='结束日期 (YYYYMMDD格式)')
    parser.add_argument('--batch_days', type=int, default=30,
                        help='批处理天数，一次处理多少天的数据')
    parser.add_argument('--force_recompute', action='store_true',
                        help='是否强制重新计算（忽略已有缓存）')
    parser.add_argument('--stock_pool', type=str, default='csi500',
                        choices=['csi500', 'csi300', 'all'],
                        help='股票池类型')
    parser.add_argument('--cache_dir', type=str, default='./factor_cache',
                        help='缓存目录路径')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("因子数据预计算脚本")
    print("=" * 80)
    print(f"开始日期: {args.start_date}")
    print(f"结束日期: {args.end_date}")
    print(f"批处理天数: {args.batch_days}")
    print(f"股票池: {args.stock_pool}")
    print(f"缓存目录: {args.cache_dir}")
    print(f"强制重新计算: {args.force_recompute}")
    print("=" * 80)
    
    # 创建FactorModel实例
    model = FactorModel(
        train_start=args.start_date,
        train_end=args.end_date,
        test_start=args.start_date,
        test_end=args.end_date,
        cache_dir=args.cache_dir
    )
    
    # 选择股票池
    db = DB()
    if args.stock_pool == 'csi500':
        stock_codes = db.get_index_stocks("000852")
        print(f"使用中证500成分股，共 {len(stock_codes)} 只股票")
    elif args.stock_pool == 'csi300':
        stock_codes = db.get_index_stocks("000905")  # 假设中证300的代码
        print(f"使用中证300成分股，共 {len(stock_codes)} 只股票")
    else:
        stock_codes = db.get_all_stock_code()
        print(f"使用全部股票，共 {len(stock_codes)} 只股票")
    
    # 开始预计算
    start_time = datetime.now()
    print(f"\n开始预计算，时间: {start_time}")
    
    try:
        model.precompute_all_factors(
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date,
            benchmark_code="sh.000001",
            batch_days=args.batch_days,
            force_recompute=args.force_recompute
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("=" * 80)
        print("预计算完成!")
        print(f"开始时间: {start_time}")
        print(f"结束时间: {end_time}")
        print(f"总耗时: {duration}")
        
        # 显示缓存统计信息
        cache_info = model.get_cache_info()
        print(f"\n缓存统计信息:")
        print(f"磁盘缓存: {cache_info['disk_cache']['size']} 个文件")
        print(f"磁盘使用: {cache_info['disk_cache']['usage_mb']:.2f} MB")
        
        # 显示预计算缓存统计
        print(f"预计算因子文件: {cache_info['precomputed_cache']['size']} 个")
        print(f"预计算因子缓存大小: {cache_info['precomputed_cache']['usage_mb']:.2f} MB")
        
        # 估算文件数量优化效果
        if cache_info['precomputed_cache']['size'] > 0:
            # 假设平均每月22个交易日
            estimated_daily_files = cache_info['precomputed_cache']['size'] * 22
            print(f"如果使用按日存储，大约需要 {estimated_daily_files} 个文件")
            print(f"文件数量减少了约 {((estimated_daily_files - cache_info['precomputed_cache']['size']) / estimated_daily_files * 100):.1f}%")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"预计算过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 