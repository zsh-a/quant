import numpy as np
from db import DB


def log2percent(x):
    return np.round(x * 100 - 100, 2)


name_cache = {}

def get_name(symbol):
    if symbol not in name_cache:
        db = DB()
        name_cache[symbol] = db.get_meta(symbol).iloc[0]["name"]

    return name_cache[symbol]
