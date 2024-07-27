#!/usr/bin/env python3
"""
统一的因子管理器
整合所有因子获取逻辑，提供简洁高效的接口
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime

# 添加项目根目录到Python路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from alpha.alphatest import AlphasTest as Alpha
from alpha.factor_analyzer import FactorAnalyzer

# 添加 jointdata 支持
try:
    sys.path.append(os.path.join(parent_dir, "jointdata"))
    from wide_table_manager import WideTableManager
    JOINTDATA_AVAILABLE = True
    logger.info("✅ jointdata模块加载成功")
except ImportError as e:
    logger.warning(f"❌ jointdata模块加载失败: {e}")
    JOINTDATA_AVAILABLE = False


class FactorManager:
    """统一的因子管理器"""
    
    def __init__(self, db_client, **config):
        """
        初始化因子管理器
        
        参数:
        - db_client: 数据库客户端
        - config: 配置参数
        """
        self.db_client = db_client
        
        # 因子源配置
        self.use_jointdata = config.get("use_jointdata", True)
        self.use_alpha_factors = config.get("use_alpha_factors", False)
        self.use_complex_factors = config.get("use_complex_factors", False)
        
        # jointdata 配置
        self.jointdata_manager = None
        self.jointdata_config = config.get("jointdata_config", {
            'host': 'localhost',
            'port': 8123,
            'user': 'default',
            'password': '',
            'database': 'factor_db',
            'table_name': 'factor_data'
        })
        
        # 复杂因子模型
        self.factor_model = None
        self.factor_analyzer = FactorAnalyzer()
        
        # 缓存配置
        self.factor_cache = {}
        self.max_cache_size = config.get("max_cache_size", 1000)
        self.max_factors = config.get("max_factors", 100)
        
        # 初始化
        self._init_jointdata()
        self._init_complex_model(config)
    
    def _init_jointdata(self):
        """初始化 jointdata 管理器"""
        if not self.use_jointdata or not JOINTDATA_AVAILABLE:
            return
            
        try:
            self.jointdata_manager = WideTableManager(
                host=self.jointdata_config['host'],
                port=self.jointdata_config['port'],
                user=self.jointdata_config['user'],
                password=self.jointdata_config['password'],
                database=self.jointdata_config['database'],
                table_name=self.jointdata_config['table_name']
            )
            logger.info(f"✅ jointdata数据库连接成功: {self.jointdata_config['database']}.{self.jointdata_config['table_name']}")
            
            # 获取可用因子列表
            available_factors = self.jointdata_manager.get_available_factors()
            logger.info(f"可用因子数量: {len(available_factors)}")
            if available_factors:
                logger.info(f"前10个因子: {available_factors[:10]}")
                
        except Exception as e:
            logger.error(f"❌ jointdata数据库连接失败: {e}")
            self.jointdata_manager = None
            self.use_jointdata = False
    
    def _init_complex_model(self, config):
        """初始化复杂因子模型"""
        if not self.use_complex_factors:
            return
            
        try:
            from alpha.factor_model import FactorModel
            self.factor_model = FactorModel()
            
            # 加载模型（如果需要）
            model_path = config.get("model_path", "./models")
            model_name = config.get("complex_model_name")
            if model_name:
                self.factor_model.load_model(model_path=model_path, model_name=model_name)
                logger.info(f"复杂因子模型加载成功: {model_name}")
                
        except Exception as e:
            logger.error(f"复杂因子模型初始化失败: {e}")
            self.factor_model = None
            self.use_complex_factors = False
    
    def get_factors(self, codes: List[str], current_date: datetime, 
                   factor_source: str = "auto") -> Optional[pd.DataFrame]:
        """
        获取因子数据
        
        参数:
        - codes: 股票代码列表
        - current_date: 当前日期
        - factor_source: 因子源 ("auto", "jointdata", "alpha", "complex")
        
        返回:
        - 因子数据DataFrame
        """
        if not codes:
            logger.warning("股票代码列表为空")
            return None
        
        # 检查缓存
        cache_key = f"{current_date}_{','.join(sorted(codes))}_{factor_source}"
        if cache_key in self.factor_cache:
            logger.debug(f"缓存命中: {cache_key[:20]}...")
            return self.factor_cache[cache_key]
        
        # 根据因子源获取数据
        factors_data = None
        
        if factor_source == "auto":
            # 自动选择因子源
            if self.use_jointdata and self.jointdata_manager:
                factors_data = self._get_jointdata_factors(codes, current_date)
            elif self.use_alpha_factors:
                factors_data = self._get_alpha_factors(codes, current_date)
            elif self.use_complex_factors and self.factor_model:
                factors_data = self._get_complex_factors(codes, current_date)
        elif factor_source == "jointdata":
            factors_data = self._get_jointdata_factors(codes, current_date)
        elif factor_source == "alpha":
            factors_data = self._get_alpha_factors(codes, current_date)
        elif factor_source == "complex":
            factors_data = self._get_complex_factors(codes, current_date)
        
        # 缓存结果
        if factors_data is not None and not factors_data.empty:
            self._add_to_cache(cache_key, factors_data)
            logger.info(f"获取因子数据成功: {len(factors_data)} 只股票, {len(factors_data.columns)} 个因子")
        
        return factors_data
    
    def _get_jointdata_factors(self, codes: List[str], current_date: datetime) -> Optional[pd.DataFrame]:
        """从 jointdata 数据库获取因子数据"""
        if not self.jointdata_manager:
            logger.warning("jointdata管理器未初始化")
            return None
        
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # 获取可用因子列表
            available_factors = self.jointdata_manager.get_available_factors()
            if not available_factors:
                logger.warning("jointdata数据库中没有可用因子")
                return None
            
            # 限制因子数量
            if len(available_factors) > self.max_factors:
                available_factors = available_factors[:self.max_factors]
                logger.info(f"使用前 {self.max_factors} 个因子")
            
            # 查询因子数据
            factors_data = self.jointdata_manager.query_cross_section(
                date=date_str,
                factors=available_factors,
                stock_codes=codes
            )
            
            if factors_data.empty:
                logger.warning(f"jointdata数据库中没有 {date_str} 的因子数据")
                return None
            
            # 设置索引
            factors_data.set_index('code', inplace=True)
            
            return factors_data
            
        except Exception as e:
            logger.error(f"从jointdata获取因子数据失败: {str(e)}")
            return None
    
    def _get_alpha_factors(self, codes: List[str], current_date: datetime) -> Optional[pd.DataFrame]:
        """获取 alpha 因子数据"""
        try:
            # 获取历史数据
            fields = ["open", "close", "low", "high", "volume", "amount", "turn", "adjfactor"]
            
            data = self.db_client.get_price(
                stocks=codes,
                end_date=str(current_date.date()),
                fields=fields,
                count=30,
                start_date=None,
            )
            
            if data.empty:
                logger.warning(f"无法获取 {codes} 在 {current_date} 的数据")
                return None
            
            # 计算VWAP
            data['vwap'] = data.groupby(level=0, group_keys=False).apply(
                lambda group: (group["volume"] * group["close"]).cumsum() / group["volume"].cumsum()
            )
            
            # 获取所有alpha因子方法
            alpha_methods = Alpha.get_alpha_methods()
            logger.info(f"发现 {len(alpha_methods)} 个alpha因子")
            
            # 计算因子
            factors_list = []
            for code, group in data.groupby(level=0):
                group_data = group.reset_index(level=0, drop=True)
                alpha = Alpha(group_data)
                
                try:
                    latest_date = group_data.index.max()
                    latest_data = group_data.loc[latest_date:latest_date].copy()
                    
                    success_count = 0
                    for method_name in alpha_methods:
                        try:
                            method = getattr(alpha, method_name)
                            factor_values = method()
                            if hasattr(factor_values, 'loc'):
                                latest_data[method_name] = factor_values.loc[latest_date:latest_date]
                            else:
                                latest_data[method_name] = factor_values
                            success_count += 1
                        except Exception as e:
                            logger.debug(f"计算股票 {code} 的因子 {method_name} 时出错: {e}")
                            latest_data[method_name] = np.nan
                            continue
                    
                    if success_count > 0:
                        latest_data.index = [code]
                        factors_list.append(latest_data)
                        
                except Exception as e:
                    logger.warning(f"计算股票 {code} 的因子时出错: {e}")
                    continue
            
            if not factors_list:
                logger.warning("没有成功计算出任何因子")
                return None
            
            # 合并所有股票的因子数据
            all_factors = pd.concat(factors_list, axis=0)
            logger.info(f"成功计算 {len(all_factors)} 只股票的alpha因子")
            
            return all_factors
            
        except Exception as e:
            logger.error(f"计算alpha因子时出错: {str(e)}")
            return None
    
    def _get_complex_factors(self, codes: List[str], current_date: datetime) -> Optional[pd.DataFrame]:
        """获取复杂因子数据"""
        if not self.factor_model:
            logger.warning("复杂因子模型未初始化")
            return None
        
        try:
            date_str = current_date.strftime('%Y%m%d')
            
            # 尝试从预计算缓存加载
            factors_data = self.factor_model.get_single_date_factors(
                stock_codes=codes,
                date=date_str,
                benchmark_code="sh.000001"
            )
            
            if factors_data is not None and not factors_data.empty:
                logger.info(f"从预计算缓存加载因子数据: {factors_data.shape}")
                return factors_data
            
            # 如果预计算数据不存在，使用实时计算
            logger.info("预计算因子数据不存在，使用实时计算...")
            
            fields = ["open", "close", "low", "high", "volume", "amount", "turn", "adjfactor"]
            
            data = self.factor_analyzer.load_data(
                codes=codes,
                end_date=str(current_date.date()),
                fields=fields,
                count=30,
                start_date=None,
            )
            
            if data.empty:
                logger.warning(f"无法获取 {codes} 在 {current_date} 的数据")
                return None
            
            # 计算因子
            self.factor_analyzer.stock_data = data
            factors = self.factor_analyzer.calculate_factors(benchmark_code="sh.000001")
            
            if factors is None or factors.empty:
                logger.warning(f"无法计算 {codes} 在 {current_date} 的因子")
                return None
            
            # 获取最新日期的因子数据
            latest_date = factors.index.get_level_values("date").max()
            latest_factors = factors.xs(latest_date, level="date")
            
            return latest_factors
            
        except Exception as e:
            logger.error(f"计算复杂因子时出错: {str(e)}")
            return None
    
    def _add_to_cache(self, cache_key: str, factors_data: pd.DataFrame):
        """添加到缓存"""
        # 限制缓存大小
        if len(self.factor_cache) >= self.max_cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.factor_cache))
            del self.factor_cache[oldest_key]
            logger.debug(f"删除缓存项: {oldest_key[:20]}...")
        
        self.factor_cache[cache_key] = factors_data
    
    def get_available_factors(self, factor_source: str = "auto") -> List[str]:
        """获取可用因子列表"""
        if factor_source == "auto":
            if self.use_jointdata and self.jointdata_manager:
                return self.jointdata_manager.get_available_factors()
            elif self.use_alpha_factors:
                return Alpha.get_alpha_methods()
            else:
                return []
        elif factor_source == "jointdata" and self.jointdata_manager:
            return self.jointdata_manager.get_available_factors()
        elif factor_source == "alpha":
            return Alpha.get_alpha_methods()
        else:
            return []
    
    def clear_cache(self):
        """清空缓存"""
        self.factor_cache.clear()
        logger.info("因子缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.factor_cache),
            "max_cache_size": self.max_cache_size,
            "cache_keys": list(self.factor_cache.keys())[:10]  # 前10个键
        } 