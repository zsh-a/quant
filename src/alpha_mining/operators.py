import pandas as pd
import numpy as np
from typing import Union

InputData = Union[pd.Series, pd.DataFrame]

# --- Helper for Logic ---
def Where(condition: InputData, x: InputData, y: InputData) -> InputData:
    """Equivalent to If-Then-Else or np.where"""
    return x.where(condition, y)

# --- Unary ---
def Abs(x: InputData) -> InputData: return x.abs()
def Log(x: InputData) -> InputData:
    """Logarithm with improved numerical stability"""
    return np.log(np.maximum(x.abs(), 1e-9))
def Sign(x: InputData) -> InputData: return np.sign(x)
def Sqrt(x: InputData) -> InputData: return np.sqrt(x.abs())

# --- Time-series ---
def Delay(x: InputData, d: int = 1) -> InputData: return x.shift(d)
def Delta(x: InputData, d: int = 1) -> InputData: return x - x.shift(d)
def Ts_Mean(x: InputData, d: int = 10) -> InputData: return x.rolling(window=d).mean()
def Ts_Sum(x: InputData, d: int = 10) -> InputData: return x.rolling(window=d).sum()
def Ts_Std(x: InputData, d: int = 10) -> InputData: return x.rolling(window=d).std()
def Ts_Max(x: InputData, d: int = 10) -> InputData: return x.rolling(window=d).max()
def Ts_Min(x: InputData, d: int = 10) -> InputData: return x.rolling(window=d).min()
def Ts_Rank(x: InputData, d: int = 10) -> InputData:
    """Optimized time-series rank using vectorized operations"""
    def rank_last(arr):
        if len(arr) < 2:
            return 0.5
        return (arr[-1] > arr[:-1]).sum() / (len(arr) - 1)
    return x.rolling(window=d).apply(rank_last, raw=True)

def Ts_Zscore(x: InputData, d: int = 10) -> InputData:
    return (x - x.rolling(window=d).mean()) / (x.rolling(window=d).std() + 1e-9)

def Ts_Returns(x: InputData, d: int = 1) -> InputData:
    return x / (x.shift(d) + 1e-9) - 1.0

def Ts_EMA(x: InputData, d: int = 10) -> InputData:
    return x.ewm(span=d, adjust=False).mean()

def Ts_DecayLinear(x: InputData, d: int = 10) -> InputData:
    weights = np.arange(1, d + 1)
    weights = weights / weights.sum()
    return x.rolling(window=d).apply(lambda s: np.dot(s, weights))

def Ts_Winsorize(x: InputData, d: int = 10, n_std: float = 3.0) -> InputData:
    """Winsorize outliers beyond n_std standard deviations"""
    mean = x.rolling(window=d).mean()
    std = x.rolling(window=d).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    return x.clip(lower=lower, upper=upper)

# --- Binary ---
def Correlation(x: InputData, y: InputData, d: int = 10) -> InputData: return x.rolling(window=d).corr(y)
def Covariance(x: InputData, y: InputData, d: int = 10) -> InputData: return x.rolling(window=d).cov(y)
def Max(x: InputData, y: InputData) -> InputData: return np.maximum(x, y)
def Min(x: InputData, y: InputData) -> InputData: return np.minimum(x, y)

# --- Cross-sectional ---
def CSRank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalization"""
    return x.rank(axis=1, pct=True)

def Scale(x: pd.DataFrame) -> pd.DataFrame:
    """Scale to sum of absolute values = 1"""
    return x.div(x.abs().sum(axis=1) + 1e-9, axis=0)

def Neutralize(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Industry neutralization (demean by group)"""
    return x.sub(x.groupby(group, axis=1).transform('mean'))

def Power(x: InputData, p: float = 2.0) -> InputData:
    """Non-linear transformation: sign(x) * |x|^p"""
    return np.sign(x) * (x.abs() ** p)

def Sigmoid(x: InputData) -> InputData:
    """Sigmoid transformation for bounded output"""
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

# --- SAFE_LOCALS with Case-Insensitive Aliases ---
SAFE_LOCALS = {
    'Abs': Abs, 'Log': Log, 'Sign': Sign, 'Sqrt': Sqrt, 'sqrt': Sqrt,
    'Delay': Delay, 'Ts_Delay': Delay, 'Delta': Delta, 'Diff': Delta,
    'Ts_Mean': Ts_Mean, 'Ts_Sum': Ts_Sum, 'Ts_Std': Ts_Std,
    'Ts_Max': Ts_Max, 'Ts_Min': Ts_Min, 'Ts_Rank': Ts_Rank,
    'Ts_Zscore': Ts_Zscore, 'Ts_ZScore': Ts_Zscore,
    'Ts_Returns': Ts_Returns, 'Ts_EMA': Ts_EMA, 'Ts_DecayLinear': Ts_DecayLinear,
    'Ts_Winsorize': Ts_Winsorize,
    'Correlation': Correlation, 'Corr': Correlation,
    'Covariance': Covariance, 'Cov': Covariance,
    'Max': Max, 'Min': Min, 'Where': Where, 'If': Where,
    'CSRank': CSRank, 'Scale': Scale, 'Neutralize': Neutralize,
    'Power': Power, 'Sigmoid': Sigmoid,
}
