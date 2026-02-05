"""
Machine Learning Integration - Feature engineering and model support for ML-based strategies.
Provides tools for feature extraction, model training, and ML strategy base class.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum

from src.core.base import Strategy, Bar
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureType(Enum):
    """Feature types for ML models"""

    PRICE = "price"  # OHLCV based
    TECHNICAL = "technical"  # Technical indicators
    FUNDAMENTAL = "fundamental"  # Fundamental data
    SENTIMENT = "sentiment"  # Sentiment data
    MACRO = "macro"  # Macro economic
    CUSTOM = "custom"


@dataclass
class FeatureConfig:
    """Configuration for a single feature"""

    name: str
    feature_type: FeatureType
    params: Dict[str, Any] = field(default_factory=dict)
    lookback: int = 1  # Number of historical values to include
    normalize: bool = True
    fill_method: str = "ffill"  # ffill, bfill, zero, mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.feature_type.value,
            "params": self.params,
            "lookback": self.lookback,
            "normalize": self.normalize,
            "fill_method": self.fill_method,
        }


class FeatureExtractor:
    """
    Feature extraction for ML models.
    Computes technical indicators and transforms raw data into features.
    """

    def __init__(self, feature_configs: List[FeatureConfig]):
        self.feature_configs = feature_configs
        self._feature_names: List[str] = []
        self._scaler_params: Dict[str, Tuple[float, float]] = {}  # mean, std

    def extract(
        self, bars_history: List[Dict[str, Bar]], fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Extract features from bar history.

        Args:
            bars_history: List of bar dictionaries (oldest first)
            fit_scaler: Whether to fit the scaler on this data

        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if not bars_history:
            return np.array([])

        # Convert to DataFrame for easier manipulation
        records = []
        for bars in bars_history:
            for symbol, bar in bars.items():
                records.append(
                    {
                        "timestamp": bar.timestamp,
                        "symbol": symbol,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "amount": bar.amount,
                    }
                )

        df = pd.DataFrame(records)
        if df.empty:
            return np.array([])

        # Extract features for each symbol
        all_features = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].sort_values("timestamp")
            features = self._extract_symbol_features(symbol_df, fit_scaler)
            all_features.append(features)

        if not all_features:
            return np.array([])

        # Stack features (last row is most recent)
        result = np.vstack(all_features)
        return result

    def _extract_symbol_features(
        self, df: pd.DataFrame, fit_scaler: bool
    ) -> np.ndarray:
        """Extract features for a single symbol"""
        features = {}

        for config in self.feature_configs:
            if config.feature_type == FeatureType.PRICE:
                values = self._extract_price_features(df, config)
            elif config.feature_type == FeatureType.TECHNICAL:
                values = self._extract_technical_features(df, config)
            else:
                values = self._extract_custom_features(df, config)

            # Apply lookback
            for lag in range(config.lookback):
                col_name = f"{config.name}_lag{lag}" if lag > 0 else config.name
                if lag > 0:
                    lagged = np.roll(values, lag)
                    lagged[:lag] = np.nan
                    features[col_name] = lagged
                else:
                    features[col_name] = values

        # Convert to DataFrame and handle missing values
        feature_df = pd.DataFrame(features)

        for col in feature_df.columns:
            config = self._get_config_for_feature(col)
            if config:
                if config.fill_method == "ffill":
                    feature_df[col] = feature_df[col].ffill()
                elif config.fill_method == "bfill":
                    feature_df[col] = feature_df[col].bfill()
                elif config.fill_method == "zero":
                    feature_df[col] = feature_df[col].fillna(0)
                elif config.fill_method == "mean":
                    feature_df[col] = feature_df[col].fillna(feature_df[col].mean())

        feature_df = feature_df.fillna(0)

        # Normalize
        if fit_scaler:
            for col in feature_df.columns:
                config = self._get_config_for_feature(col)
                if config and config.normalize:
                    mean = feature_df[col].mean()
                    std = feature_df[col].std()
                    if std > 0:
                        self._scaler_params[col] = (mean, std)
                        feature_df[col] = (feature_df[col] - mean) / std
        else:
            for col in feature_df.columns:
                if col in self._scaler_params:
                    mean, std = self._scaler_params[col]
                    if std > 0:
                        feature_df[col] = (feature_df[col] - mean) / std

        self._feature_names = list(feature_df.columns)
        return feature_df.values

    def _get_config_for_feature(self, feature_name: str) -> Optional[FeatureConfig]:
        """Get config for a feature name (handles lag suffix)"""
        base_name = feature_name.split("_lag")[0]
        for config in self.feature_configs:
            if config.name == base_name:
                return config
        return None

    def _extract_price_features(
        self, df: pd.DataFrame, config: FeatureConfig
    ) -> np.ndarray:
        """Extract price-based features"""
        name = config.name
        params = config.params

        if name == "returns":
            period = params.get("period", 1)
            return df["close"].pct_change(period).values

        elif name == "log_returns":
            period = params.get("period", 1)
            return np.log(df["close"] / df["close"].shift(period)).values

        elif name == "volatility":
            period = params.get("period", 20)
            return df["close"].pct_change().rolling(period).std().values

        elif name == "volume_ratio":
            period = params.get("period", 20)
            return (df["volume"] / df["volume"].rolling(period).mean()).values

        elif name == "range":
            return ((df["high"] - df["low"]) / df["close"]).values

        elif name == "gap":
            return ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)).values

        else:
            # Default: return close price
            return df["close"].values

    def _extract_technical_features(
        self, df: pd.DataFrame, config: FeatureConfig
    ) -> np.ndarray:
        """Extract technical indicator features"""
        name = config.name
        params = config.params
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        if name == "sma":
            period = params.get("period", 20)
            sma = close.rolling(period).mean()
            return ((close - sma) / sma).values  # Distance from SMA

        elif name == "ema":
            period = params.get("period", 20)
            ema = close.ewm(span=period, adjust=False).mean()
            return ((close - ema) / ema).values

        elif name == "rsi":
            period = params.get("period", 14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return (rsi.fillna(50) / 100).values  # Normalize to 0-1

        elif name == "macd":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)

            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()

            return ((macd_line - signal_line) / close).values

        elif name == "bollinger":
            period = params.get("period", 20)
            std_mult = params.get("std", 2)

            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper = sma + std_mult * std
            lower = sma - std_mult * std

            # Return position within bands (0-1)
            band_width = upper - lower
            position = (close - lower) / band_width.replace(0, np.nan)
            return position.fillna(0.5).values

        elif name == "atr":
            period = params.get("period", 14)

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()

            return (atr / close).values  # Normalize by price

        elif name == "obv":
            # On-Balance Volume
            direction = np.sign(close.diff())
            obv = (direction * volume).cumsum()
            obv_sma = obv.rolling(20).mean()
            return ((obv - obv_sma) / obv_sma.abs().replace(0, 1)).values

        else:
            return np.zeros(len(df))

    def _extract_custom_features(
        self, df: pd.DataFrame, config: FeatureConfig
    ) -> np.ndarray:
        """Extract custom features (placeholder for extension)"""
        return np.zeros(len(df))

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Get number of features"""
        return len(self._feature_names)


@dataclass
class ModelConfig:
    """Configuration for ML model"""

    model_type: str  # linear, tree, nn, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    target: str = "returns"  # Target variable
    target_horizon: int = 1  # Prediction horizon in bars
    train_window: int = 252  # Training window size
    retrain_freq: int = 20  # Retrain every N bars


class MLModelWrapper:
    """
    Wrapper for ML models with common interface.
    Supports sklearn-compatible models.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_fitted = False

        self._create_model()

    def _create_model(self):
        """Create the ML model based on config"""
        model_type = self.config.model_type
        params = self.config.params

        try:
            if model_type == "linear":
                from sklearn.linear_model import Ridge

                self.model = Ridge(**params)

            elif model_type == "lasso":
                from sklearn.linear_model import Lasso

                self.model = Lasso(**params)

            elif model_type == "elastic_net":
                from sklearn.linear_model import ElasticNet

                self.model = ElasticNet(**params)

            elif model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor

                self.model = RandomForestRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 10),
                    random_state=42,
                    n_jobs=-1,
                )

            elif model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor

                self.model = GradientBoostingRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 5),
                    learning_rate=params.get("learning_rate", 0.1),
                    random_state=42,
                )

            elif model_type == "xgboost":
                try:
                    import xgboost as xgb

                    self.model = xgb.XGBRegressor(
                        n_estimators=params.get("n_estimators", 100),
                        max_depth=params.get("max_depth", 6),
                        learning_rate=params.get("learning_rate", 0.1),
                        random_state=42,
                        n_jobs=-1,
                    )
                except ImportError:
                    logger.warning(
                        "XGBoost not installed, falling back to GradientBoosting"
                    )
                    from sklearn.ensemble import GradientBoostingRegressor

                    self.model = GradientBoostingRegressor()

            else:
                # Default: Ridge regression
                from sklearn.linear_model import Ridge

                self.model = Ridge()

        except ImportError as e:
            logger.error(f"Failed to import ML model: {e}")
            raise

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model"""
        if len(X) < 10:
            logger.warning("Not enough samples to train model")
            return

        # Handle NaN/Inf
        mask = ~(
            np.isnan(X).any(axis=1)
            | np.isnan(y)
            | np.isinf(X).any(axis=1)
            | np.isinf(y)
        )
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 10:
            logger.warning("Not enough clean samples after removing NaN/Inf")
            return

        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        logger.info(f"Model fitted on {len(X_clean)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            return np.zeros(len(X))

        # Handle NaN/Inf in input
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return self.model.predict(X_clean)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance (for tree-based models)"""
        if not self.is_fitted:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))

        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_)
            if coefs.sum() > 0:
                coefs = coefs / coefs.sum()
            return dict(zip(feature_names, coefs))

        return {}


class MLStrategy(Strategy):
    """
    Base class for ML-based trading strategies.
    Handles feature extraction, model training, and prediction.
    """

    def __init__(
        self,
        db_client,
        session_id: Optional[str] = None,
        feature_configs: Optional[List[FeatureConfig]] = None,
        model_config: Optional[ModelConfig] = None,
        **params,
    ):
        super().__init__(session_id)
        self.db_client = db_client
        self.params = params

        # Default feature configs
        if feature_configs is None:
            feature_configs = [
                FeatureConfig("returns", FeatureType.PRICE, {"period": 1}),
                FeatureConfig("returns", FeatureType.PRICE, {"period": 5}, lookback=5),
                FeatureConfig("volatility", FeatureType.PRICE, {"period": 20}),
                FeatureConfig("rsi", FeatureType.TECHNICAL, {"period": 14}),
                FeatureConfig("macd", FeatureType.TECHNICAL),
                FeatureConfig("bollinger", FeatureType.TECHNICAL, {"period": 20}),
            ]

        # Default model config
        if model_config is None:
            model_config = ModelConfig(
                model_type="random_forest",
                target="returns",
                target_horizon=1,
                train_window=252,
                retrain_freq=20,
            )

        self.feature_extractor = FeatureExtractor(feature_configs)
        self.model = MLModelWrapper(model_config)
        self.model_config = model_config

        # State
        self._bars_history: List[Dict[str, Bar]] = []
        self._bar_count = 0
        self._last_train_bar = 0

    def on_bar(self, bars: Dict[str, Bar]):
        """Process new bar data"""
        self._bars_history.append(bars)
        self._bar_count += 1

        # Limit history size
        max_history = self.model_config.train_window + 100
        if len(self._bars_history) > max_history:
            self._bars_history = self._bars_history[-max_history:]

        # Check if we need to retrain
        if (self._bar_count - self._last_train_bar) >= self.model_config.retrain_freq:
            self._train_model()
            self._last_train_bar = self._bar_count

        # Make predictions and trade
        if self.model.is_fitted:
            predictions = self._predict(bars)
            self._execute_trades(bars, predictions)

    def _train_model(self):
        """Train the model on historical data"""
        if len(self._bars_history) < self.model_config.train_window:
            return

        # Extract features
        X = self.feature_extractor.extract(
            self._bars_history[-self.model_config.train_window :], fit_scaler=True
        )

        if len(X) < 50:
            return

        # Create target (future returns)
        horizon = self.model_config.target_horizon
        y = self._create_target(
            self._bars_history[-self.model_config.train_window :], horizon
        )

        # Align X and y (remove last 'horizon' rows from X, first 'horizon' from y)
        X = X[:-horizon]
        y = y[horizon:]

        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

        self.model.fit(X, y)

        # Log feature importance
        importance = self.model.get_feature_importance(
            self.feature_extractor.feature_names
        )
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            self._log(f"Top features: {top_features}", level="DEBUG")

    def _create_target(
        self, bars_history: List[Dict[str, Bar]], horizon: int
    ) -> np.ndarray:
        """Create target variable (future returns)"""
        if not bars_history:
            return np.array([])

        # Get close prices for first symbol
        symbol = next(iter(bars_history[0].keys()))
        closes = [
            bars.get(symbol, Bar(symbol, datetime.now(), 0, 0, 0, 0, 0, 0)).close
            for bars in bars_history
        ]

        closes = np.array(closes)
        returns = np.zeros(len(closes))
        returns[:-horizon] = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]

        return returns

    def _predict(self, bars: Dict[str, Bar]) -> Dict[str, float]:
        """Make predictions for current bars"""
        # Use recent history for features
        recent = (
            self._bars_history[-50:]
            if len(self._bars_history) >= 50
            else self._bars_history
        )
        X = self.feature_extractor.extract(recent, fit_scaler=False)

        if len(X) == 0:
            return {}

        # Predict on last row
        pred = self.model.predict(X[-1:])

        # Return prediction for each symbol
        predictions = {}
        for symbol in bars.keys():
            predictions[symbol] = float(pred[0]) if len(pred) > 0 else 0.0

        return predictions

    @abstractmethod
    def _execute_trades(self, bars: Dict[str, Bar], predictions: Dict[str, float]):
        """Execute trades based on predictions. Override in subclass."""
        pass

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema"""
        return {
            "model_type": {
                "type": "str",
                "default": "random_forest",
                "description": "ML model type",
                "options": [
                    "linear",
                    "lasso",
                    "random_forest",
                    "gradient_boosting",
                    "xgboost",
                ],
            },
            "train_window": {
                "type": "int",
                "default": 252,
                "description": "Training window size (bars)",
                "min": 50,
                "max": 1000,
            },
            "retrain_freq": {
                "type": "int",
                "default": 20,
                "description": "Retrain frequency (bars)",
                "min": 1,
                "max": 100,
            },
            "prediction_threshold": {
                "type": "float",
                "default": 0.001,
                "description": "Minimum predicted return to trade",
                "min": 0.0,
                "max": 0.1,
            },
        }
