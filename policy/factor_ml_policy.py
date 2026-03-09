import os
import sys
from loguru import logger
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from typing import Optional

from .base_policy import OrderPolicy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)
import global_var

from account import Account
from market_env import MultiMarketEnv
from order import Order, OrderManager
from alpha.factor_model import FactorModel
from alpha.simple_factor_model import SimpleFactorModel
from alpha.factor_analyzer import FactorAnalyzer
from alpha.alphatest import AlphasTest as Alpha
import talib as ta
import utils.utils as util

# 导入统一的因子管理器
from .factor_manager import FactorManager

PRICE_CHANGE_LIMIT = 0.098
MAX_POSITION = 99999999


class OrderPolicy(OrderPolicy):
    """简化的订单策略，只负责订单执行，不包含决策逻辑"""

    def __init__(self, account, **args) -> None:
        self.last_obs = []
        self.cur_obs = None

        self.account: Account = account

        self.db_client = args["db_client"]
        self.slip = 0.002  # 改为百分比滑点，0.002表示0.2%
        self.tracking = []

        self.running_in_day = False
        self.current_date = None

    def order_callback(self, order: Order, order_manager: OrderManager):
        pass

    def get_now_price(self, code, key):
        """获取当前价格"""
        if self.current_date is None:
            logger.error("current_date is None")
            return 0
            
        df = self.db_client.get_price(
            [code],
            str(self.current_date.date()),
            [key],
            1,
            str(self.current_date.date()),
        )
        if len(df) == 0:
            logger.error(
                f"can not find {code} price, current_date : {self.current_date}"
            )
            return 0

        df.reset_index(level="date", drop=True, inplace=True)
        open_price = df.loc[code, key]
        return open_price

    def check_up_down_limit(self, code, key):
        """检查涨跌停"""
        if self.current_date is None:
            logger.error("current_date is None")
            return False
            
        df = self.db_client.get_price(
            code, str(self.current_date.date()), ["open", "close"], 2
        )
        df.reset_index(level="code", drop=True, inplace=True)
        if len(df) == 0 or df.index[-1] != self.current_date:
            logger.error(f"{code} 停牌 {self.current_date}")
            return False

        today_price = df.iloc[-1][key]
        prev_close = df.iloc[0]["close"]

        if today_price / prev_close - 1 < -PRICE_CHANGE_LIMIT:
            logger.error(f"{code} 跌停 {self.current_date}")
            return False

        if today_price / prev_close - 1 > PRICE_CHANGE_LIMIT:
            logger.error(f"{code} 涨停 {self.current_date}")
            return False
        return True

    def buy_policy(self, order: Order):
        """买入策略执行 - 只负责执行，不包含决策逻辑"""
        if not self.check_up_down_limit(order.symbol, order.exec_time):
            return (False, 0)

        trading_price = self.get_now_price(order.symbol, order.exec_time) * (
            1 + self.slip
        )

        action = order.quantity
        min_action = self.account.min_action

        if action > min_action:
            action = action // min_action * min_action
            num_stakes = min(
                self.account.cash // trading_price // min_action * min_action,
                action,
            )

            if num_stakes >= min_action and num_stakes > order.quantity * 0.5:
                amount = trading_price * num_stakes
                cost = amount * self.account.trading_fee_open

                while num_stakes > 0 and self.account.cash < amount + cost:
                    num_stakes -= min_action
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_fee_open

                if num_stakes > 0:
                    order.quantity = num_stakes
                    logger.info(
                        f"执行买入: {order.symbol}, 数量: {num_stakes}, 价格: {trading_price}"
                    )
                    return (True, trading_price)

        return (False, trading_price)

    def buy_cond(self, code):
        """买入条件判断 - 简化为总是返回True，由Agent决定下单"""
        return True

    def sell_policy(self, order):
        """卖出策略执行 - 只负责执行，不包含决策逻辑"""
        action = order.quantity

        if not self.check_up_down_limit(order.symbol, order.exec_time):
            return (False, 0)

        trading_price = self.get_now_price(order.symbol, order.exec_time) * (
            1 - self.slip
        )
        min_action = self.account.min_action
        action = action // min_action * min_action

        num_stakes = min(self.account.positions[-1][order.symbol].quantity, action)
        if num_stakes > 0:
            order.quantity = num_stakes
            logger.info(
                f"执行卖出: {order.symbol}, 数量: {num_stakes}, 价格: {trading_price}"
            )
            return (True, trading_price)

        return (False, trading_price)

    def sell_cond(self, code):
        """卖出条件判断 - 简化为总是返回True，由Agent决定下单"""
        return True

    def step(self, obs):
        """策略步进"""
        if self.cur_obs:
            self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs
        self.current_date = obs[0].name

    def step_in_day(self, obs):
        """日内步进"""
        self.cur_obs = obs
        self.running_in_day = True


class Agent:
    """基于二分类因子模型的智能交易代理 - 负责所有下单决策逻辑"""

    def __init__(self, market_env: MultiMarketEnv, **args) -> None:
        self.market_env = market_env
        self.db_client = args["db_client"]

        # 统一的因子管理器
        self.factor_manager = FactorManager(self.db_client)
        
        # 模型相关
        self.simple_factor_model: Optional[SimpleFactorModel] = None
        self.model_loaded = False
        self.use_simple_model = args.get("use_simple_model", True)
        # 是否使用分阶段模型
        self.use_regime_models = args.get("use_regime_models", True)
        self.regime_bundle_name = args.get("regime_bundle_name", "my_regime_202408")
        self.regime_bundle_file = args.get("regime_bundle_file", "./simple_models/my_regime_202408_bundle.json")

        # 模型预测结果缓存
        self.prediction_cache = {}

        self.current_date = None
        self.max_stocks = args.get("max_stocks", 5)  # 最大持股数量
        self.prediction_threshold = args.get("prediction_threshold", 0.0)  # 预测阈值
        self.sell_threshold = args.get("sell_threshold", 0.4)  # 卖出阈值

        # 加载交易日历
        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )

        # 加载模型
        model_path = args.get("model_path", "./simple_models")
        model_name = args.get("model_name", "simple_factor_model_20250816_165105")
        if model_name:
            # 根据模型文件扩展名自动选择加载方法
            if model_name.endswith('.txt'):
                self.load_lgb_model(model_path, model_name.replace('.txt', ''))
            elif self.use_simple_model:
                self.load_simple_factor_model(model_path, model_name)
            else:
                self.load_factor_model(model_path, model_name)

        self.init_indicators()

    def init_indicators(self):
        self.market_env.clean_data()

    def load_factor_model(self, model_path, model_name):
        """加载复杂因子模型"""
        try:
            from alpha.factor_model import FactorModel
            self.factor_model = FactorModel()
            model_info = self.factor_model.load_model(
                model_path=model_path, model_name=model_name
            )
            self.model_loaded = True
            logger.info(f"复杂因子模型加载成功: {model_name}")
            if model_info:
                logger.info(
                    f"模型训练期间: {model_info.get('train_period', 'Unknown')}"
                )
                logger.info(f"特征数量: {model_info.get('feature_count', 'Unknown')}")
        except Exception as e:
            logger.error(f"加载复杂因子模型失败: {str(e)}")
            self.model_loaded = False

    def load_simple_factor_model(self, model_path, model_name):
        """加载简化因子模型"""
        try:
            self.simple_factor_model = SimpleFactorModel()
            model_info = self.simple_factor_model.load_model(
                model_path=model_path, model_name=model_name
            )
            self.model_loaded = True
            logger.info(f"简化因子模型加载成功: {model_name}")
            if model_info:
                logger.info(
                    f"模型训练期间: {model_info.get('train_start', 'Unknown')} - {model_info.get('train_end', 'Unknown')}"
                )
                logger.info(f"特征数量: {model_info.get('feature_count', 'Unknown')}")
                
                # 保存模型的特征信息
                feature_names = model_info.get('feature_names', [])
                self.simple_factor_model.feature_names = feature_names
                logger.info(f"模型特征列表: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
                
                # 如果有特征预处理信息，也保存下来
                if 'feature_preprocessing' in model_info:
                    self.simple_factor_model.feature_preprocessing_info = model_info['feature_preprocessing']
                    preprocessing_info = model_info['feature_preprocessing']
                    kept_features = preprocessing_info.get('kept_features', [])
                    logger.info(f"特征预处理后保留的特征: {kept_features[:10]}{'...' if len(kept_features) > 10 else ''}")

            # 加载分阶段模型bundle（可选）
            if self.use_regime_models and (self.regime_bundle_file or self.regime_bundle_name):
                try:
                    meta = self.simple_factor_model.load_regime_models(
                        model_path=model_path,
                        bundle_name=self.regime_bundle_name,
                        bundle_file=self.regime_bundle_file,
                    )
                    logger.info(
                        f"分阶段模型bundle加载成功: {meta.get('bundle_name', 'unknown')}，包含 {len(meta.get('model_files', {}))} 个子模型"
                    )
                except Exception as e:
                    logger.error(f"加载分阶段模型bundle失败: {str(e)}")
        except Exception as e:
            logger.error(f"加载简化因子模型失败: {str(e)}")
            self.model_loaded = False

    def load_lgb_model(self, model_path, model_name):
        """加载LGB模型"""
        try:
            # 创建一个SimpleFactorModel实例来加载模型
            self.simple_factor_model = SimpleFactorModel()
            
            # 构造模型文件路径
            model_file = os.path.join(model_path, f"{model_name}.txt")
            
            # 加载模型
            booster = lgb.Booster(model_file=model_file)
            
            # 创建LGBMClassifier包装器
            self.simple_factor_model.model = LGBMClassifier()
            self.simple_factor_model.model._Booster = booster
            
            self.model_loaded = True
            logger.info(f"LGB模型加载成功: {model_name}")
        except Exception as e:
            logger.error(f"加载LGB模型失败: {str(e)}")
            self.model_loaded = False

    def get_next_trading_day(self):
        """获取下一个交易日"""
        date = pd.to_datetime(self.current_date)
        next_date = date + pd.DateOffset(days=1)
        while self.trad_days.loc[next_date, "is_trading_day"] == 0:
            next_date += pd.DateOffset(days=1)
        return next_date

    def step(self):
        """代理步进"""
        self.current_date = self.market_env.cur_date

    def select_stock_pool(self, end_date):
        """选择股票池"""
        # 获取中证500成分股
        stocks = self.db_client.get_index_stocks("000985", end_date)
        return stocks

    def filter_basic(self, stocks):
        """基础股票过滤"""
        if self.current_date is None:
            logger.error("current_date is None")
            return []
            
        df = self.db_client.get_price(
            stocks, str(self.current_date.date()), ["isST"], 1
        )
        df.reset_index(level="date", drop=True, inplace=True)
        df = df[(df["tradestatus"] == 1) & (df["isST"] == 0)]
        return df.index.to_list()

    def get_stock_factors(self, codes, current_date):
        """获取股票的因子数据 - 使用统一的因子管理器"""
        if not self.model_loaded:
            logger.warning("因子模型未加载，无法计算因子")
            return None

        # 使用统一的因子管理器获取因子
        return self.factor_manager.get_factors(codes, current_date)

    def predict_stock_probabilities(self, stocks, current_date):
        """批量预测股票属于高于等于平均收益组的概率"""
        if not self.model_loaded:
            logger.warning("因子模型未加载，无法进行预测")
            return {}

        # 检查预测缓存
        cache_key = f"{current_date}_{','.join(sorted(stocks))}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # 获取因子数据
        factors_data = self.get_stock_factors(stocks, current_date)
        if factors_data is None:
            return {}

        try:
            if self.simple_factor_model:
                # 获取模型训练时使用的特征列表
                model_features = self.get_model_features()
                
                if not model_features:
                    logger.warning("无法获取模型特征列表，尝试使用所有可用因子")
                    # 回退到原来的逻辑
                    available_factors = self.factor_manager.get_available_factors()
                    model_features = [
                        col for col in available_factors if col in factors_data.columns
                    ]
                
                # 过滤出实际存在的因子
                available_factors = [
                    col for col in model_features if col in factors_data.columns
                ]
                
                if not available_factors:
                    logger.warning("没有可用的因子进行预测")
                    return {}

                # 检查是否有缺失的因子
                missing_factors = [col for col in model_features if col not in factors_data.columns]
                if missing_factors:
                    logger.warning(f"缺失以下因子: {missing_factors[:10]}{'...' if len(missing_factors) > 10 else ''}")

                logger.info(f"模型特征数量: {len(model_features)}, 可用因子数量: {len(available_factors)}")

                # 准备预测数据，确保特征顺序与训练时一致
                X_pred = factors_data[available_factors].fillna(0)
                
                # 如果模型有特征预处理信息，应用相同的预处理
                if hasattr(self.simple_factor_model, 'feature_preprocessing_info'):
                    preprocessing_info = self.simple_factor_model.feature_preprocessing_info
                    kept_features = preprocessing_info.get('kept_features', [])
                    # 只使用预处理后保留的特征
                    available_factors = [col for col in available_factors if col in kept_features]
                    X_pred = factors_data[available_factors].fillna(0)
                    logger.info(f"应用特征预处理后，使用 {len(available_factors)} 个因子")

                # 进行二分类预测（优先使用分阶段加权模型）
                as_of_date_str = pd.to_datetime(current_date).strftime('%Y%m%d')
                if getattr(self.simple_factor_model, 'regime_models', None):
                    predictions = self.simple_factor_model.predict_highest_group_proba_regime_weighted(
                        X_pred,
                        as_of_date=as_of_date_str,
                        lookback_days=60,
                    )
                else:
                    predictions = self.simple_factor_model.predict_highest_group_proba(X_pred)
            else:
                logger.warning("模型未初始化")
                return {}

            # 构造结果字典
            result = {}
            for i, code in enumerate(X_pred.index):
                if i < len(predictions):
                    result[code] = predictions[i]

            # 缓存结果
            self.prediction_cache[cache_key] = result

            logger.info(f"成功预测 {len(result)} 只股票属于高于等于平均收益组的概率")
            return result

        except Exception as e:
            logger.error(f"批量预测股票概率时出错: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}

    def should_buy(self, code, probability):
        """买入决策逻辑 - 基于二分类模型预测"""
        if not self.model_loaded:
            return False

        # 检查最大持股数量限制
        if hasattr(self.market_env.account, 'get_position_price'):
            current_positions = self.market_env.account.get_position_price(
                str(self.current_date.date())
            )
            if len(current_positions) >= self.max_stocks:
                return False

        # 检查是否已经持有
        if hasattr(self.market_env.account, 'positions') and self.market_env.account.positions:
            if code in self.market_env.account.positions[-1]:
                return False

        # 检查预测概率是否超过阈值（预测股票属于高于等于平均收益组的概率）
        buy_signal = probability > self.prediction_threshold

        if buy_signal:
            logger.info(
                f"买入决策: {code}, 高于平均收益概率: {probability:.4f}, 阈值: {self.prediction_threshold}"
            )

        return buy_signal

    def should_sell(self, code, probability):
        """卖出决策逻辑 - 基于二分类模型预测"""
        if not self.model_loaded:
            return True  # 如果没有模型，使用简单策略

        # 检查是否持有
        if hasattr(self.market_env.account, 'positions') and self.market_env.account.positions:
            if code not in self.market_env.account.positions[-1]:
                return False

            if hasattr(self.market_env.account, 'availables') and self.market_env.account.availables:
                if not self.market_env.account.availables[-1][code]:
                    return False

        # 如果预测概率较低，则卖出（预测股票属于高于等于平均收益组的概率较低）
        sell_signal = probability < self.sell_threshold

        if sell_signal:
            logger.info(
                f"卖出决策: {code}, 高于平均收益概率: {probability:.4f}, 阈值: {self.sell_threshold}"
            )

        return sell_signal

    def action_decider(self, stocks_obs):
        """交易决策 - 基于二分类模型预测进行每日调仓"""
        if not self.model_loaded:
            logger.warning("因子模型未加载，跳过本次决策")
            return
        ts = self.market_env.cur_date
        today = str(ts.date())

        if self.trad_days.loc[today, "is_last_trading_day"] == 0:
            return

        stocks = self.select_stock_pool(today)

        factors = self.get_stock_factors(stocks, self.current_date)
        if factors is not None and 'tradestatus' in factors.columns and 'isST' in factors.columns:
            factors = factors[(factors["tradestatus"] == 1) & (factors["isST"] == 0)]
            valid_stocks = factors.index.tolist()
        else:
            # 如果没有交易状态信息，使用原始股票列表
            valid_stocks = stocks
            
        predictions = self.predict_stock_probabilities(valid_stocks, self.current_date)
        # 选择高预测概率的股票（预测属于高于等于平均收益组概率较高的股票）
        high_prob_stocks = [
            (code, prob)
            for code, prob in predictions.items()
            if prob > self.prediction_threshold
        ]

        # 按预测概率排序
        high_prob_stocks.sort(key=lambda x: x[1], reverse=True)

        # 选择前N只股票
        target_stocks = [code for code, _ in high_prob_stocks[: self.max_stocks]]

        logger.info(f"选中目标股票（高于平均收益概率较高）: {high_prob_stocks[:self.max_stocks]}")

        self.adjust(target_stocks)

    def get_current_date_str(self):
        return str(self.current_date.date())
        
    def process_order_value(self, target_value):
        pos_df = self.market_env.account.get_position_price(self.get_current_date_str())
        if len(pos_df) > 0:
            pos_df["value"] = pos_df["position"] * pos_df["close"]

        new_price = self.db_client.get_price(
            list(target_value.keys()), self.get_current_date_str(), ["close"], 1
        )
        new_price.reset_index(level="date", drop=True, inplace=True)
        action_pos = {}
        for code, value in target_value.items():
            if code in pos_df.index:
                new_pos = int(value / pos_df.loc[code, "close"])
                action_pos[code] = new_pos - pos_df.loc[code, "position"]
            else:
                action_pos[code] = int(value / new_price.loc[code, "close"])

        for code, action in action_pos.items():
            self.create_order(code, action)

    def adjust(self, stocks):
        target = stocks[: min(len(stocks), self.max_stocks)]
        hold_list = list(self.market_env.account.positions[-1].keys())
        target_value = {}
        for stock in hold_list:
            if stock not in target:
                target_value[stock] = 0

        total_value = self.market_env.account.get_total_value()
        for code in target:
            if code not in hold_list:
                target_value[code] = total_value / len(target)

        if len(target_value) > 0:
            logger.info(f"target_value : {target_value}")
            self.process_order_value(target_value)

    def monthly_rebalance(self):
        """月度再平衡 - 基于二分类模型预测"""
        today = str(self.current_date.date())

        # 获取股票池
        stock_pool = self.select_stock_pool(today)

        # 预测股票概率
        predictions = self.predict_stock_probabilities(stock_pool, self.current_date)

        if not predictions:
            logger.warning("没有预测结果，跳过本次再平衡")
            return

        # 选择高预测概率的股票（预测属于高于等于平均收益组概率较高的股票）
        high_prob_stocks = [
            (code, prob)
            for code, prob in predictions.items()
            if prob > self.prediction_threshold
        ]

        # 按预测概率排序
        high_prob_stocks.sort(key=lambda x: x[1], reverse=True)

        # 选择前N只股票
        target_stocks = [code for code, _ in high_prob_stocks[: self.max_stocks]]

        logger.info(f"月度再平衡 - 选中目标股票（高于平均收益概率较高）: {target_stocks}")
        for code, prob in high_prob_stocks[: self.max_stocks]:
            logger.info(f"  {code}: 高于平均收益概率 {prob:.4f}")

        # 执行调仓
        self.rebalance_portfolio(target_stocks)

    def rebalance_portfolio(self, target_stocks):
        """组合再平衡"""
        current_positions = list(self.market_env.account.positions[-1].keys())

        # 卖出不在目标列表中的股票
        for stock in current_positions:
            if stock not in target_stocks:
                self.create_order(stock, -MAX_POSITION)
                logger.info(f"再平衡卖出: {stock}")

        # 买入目标股票（等权重）
        if target_stocks:
            total_value = self.market_env.account.get_total_value()
            weight_per_stock = 1.0 / len(target_stocks)

            for stock in target_stocks:
                if stock not in current_positions:
                    target_value = total_value * weight_per_stock
                    self.create_order_by_value(stock, target_value)
                    logger.info(f"再平衡买入: {stock}, 目标价值: {target_value:.2f}")

    def create_order(self, code, action, exec_time="open"):
        """创建订单"""
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(
                code, "sell", abs(action), None, exec_time
            )
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(
                code, "buy", abs(action), None, exec_time
            )

    def create_order_by_value(self, code, target_value, exec_time="open"):
        """根据目标价值创建订单"""
        try:
            # 获取当前价格
            price_data = self.db_client.get_price(
                [code], str(self.current_date.date()), ["close"], 1
            )
            if len(price_data) == 0:
                logger.error(f"无法获取 {code} 的价格")
                return

            price_data.reset_index(level="date", drop=True, inplace=True)
            current_price = price_data.loc[code, "close"]

            # 计算目标股数
            target_shares = int(target_value / current_price)

            if target_shares > self.market_env.min_action:
                self.create_order(code, target_shares, exec_time)

        except Exception as e:
            logger.error(f"创建价值订单时出错: {str(e)}")

    def run_end(self):
        """交易日结束处理"""
        pass

    def get_model_features(self):
        """获取模型训练时使用的特征列表"""
        if not self.model_loaded:
            return []
            
        try:
            if self.simple_factor_model:
                # 方法1: 从模型的特征名称获取
                if hasattr(self.simple_factor_model, 'train_X') and self.simple_factor_model.train_X is not None:
                    return list(self.simple_factor_model.train_X.columns)
                
                # 方法2: 从模型的特征重要性获取
                if hasattr(self.simple_factor_model, 'model') and self.simple_factor_model.model is not None:
                    if hasattr(self.simple_factor_model.model, 'feature_name_'):
                        return self.simple_factor_model.model.feature_name_
                
                # 方法3: 从保存的模型信息获取（通过动态属性）
                if hasattr(self.simple_factor_model, 'feature_names'):
                    return getattr(self.simple_factor_model, 'feature_names', [])
                
                # 方法4: 从特征预处理信息获取
                if hasattr(self.simple_factor_model, 'feature_preprocessing_info'):
                    preprocessing_info = self.simple_factor_model.feature_preprocessing_info
                    return preprocessing_info.get('kept_features', [])
                
                logger.warning("无法从简单因子模型获取特征列表")
                return []
                
            elif hasattr(self, 'factor_model') and self.factor_model:
                # 复杂因子模型的特征获取逻辑
                if hasattr(self.factor_model, 'feature_names'):
                    return self.factor_model.feature_names
                logger.warning("无法从复杂因子模型获取特征列表")
                return []
                
        except Exception as e:
            logger.error(f"获取模型特征列表时出错: {str(e)}")
            return []
        
        return []


if __name__ == "__main__":
    # 示例用法
    from src.market_data.db import DB

    # 初始化参数
    db_client = DB()
    args = {
        "db_client": db_client,
        "model_path": "./simple_models",
        "model_name": "simple_factor_model_20250608_183915",  # 使用简化因子模型
        "use_simple_model": True,  # 使用简化模型
        "use_jointdata": True,  # 启用 jointdata 支持
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
        "max_stocks": 6,
        "stock_pool": 50,
    }

    # 创建因子ML代理
    # agent = Agent(None, **args)
    print("因子ML策略模块已创建")
