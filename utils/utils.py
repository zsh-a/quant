import numpy as np
import db


def log2percent(x):
    return np.round(x * 100 - 100, 2)


name_cache = {}


def get_name(symbol):
    if symbol not in name_cache:
        name_cache[symbol] = db.get_meta(symbol).iloc[0]["name"]

    return name_cache[symbol]
