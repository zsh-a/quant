import requests
import json
from typing import Union, Optional


BASE_URL = "http://172.19.112.1:11122"


def balance():
    try:
        resp = requests.get(f"{BASE_URL}/balance")
    except Exception as e:
        return str(e)
    return json.loads(resp.text)


def position():
    try:
        resp = requests.get(f"{BASE_URL}/position")
    except Exception as e:
        return str(e)
    return json.loads(resp.text)


def buy(code: str, amount: Optional[int] = None, price: Optional[float] = None):
    if price and amount % 100 != 0:
        return {"status": -1, "data": "amount must be times of 100"}
    url = f"{BASE_URL}/buy/?code={code}"

    if amount is not None:
        url += f"&amount={amount}"
    if price is not None:
        url += f"&price={price}"

    try:
        resp = requests.get(url)
    except Exception as e:
        return str(e)
    return json.loads(resp.text)


def sell(code: str, amount: Optional[int] = None, price: Optional[float] = None):
    if price and amount % 100 != 0:
        return {"status": -1, "data": "amount must be times of 100"}

    url = f"{BASE_URL}/sell/?code={code}"

    if amount is not None:
        url += f"&amount={amount}"
    if price is not None:
        url += f"&price={price}"

    try:
        resp = requests.get(url)
    except Exception as e:
        return str(e)
    return json.loads(resp.text)


# import jqktrader

# user = jqktrader.use()
# user.connect(
#     exe_path=r'D:\ths\xiadan.exe',
#     tesseract_cmd=r'tesseract'
# )


if __name__ == "__main__":
    # print(buy('510050',100))
    # print(sell('513060',100))
    # print(sell("513050"))
    print(balance())
    # try:
    #    print(user.buy('510880',None,None))
    # except Exception as e:
    #    print(e)
