import requests
import json
from typing import Union, Optional
import utils.feichu as msg


class TraderClient:
    # BASE_URL = "http://172.19.112.1:11122"

    def __init__(self, url) -> None:
        self.server_url = url

    def balance(self):
        try:
            resp = requests.get(f"{self.server_url}/balance")
        except Exception as e:
            return str(e)
        return json.loads(resp.text)

    def position(self):
        try:
            resp = requests.get(f"{self.server_url}/position")
        except Exception as e:
            return str(e)
        return json.loads(resp.text)

    def buy(
        self, code: str, amount: Optional[int] = None, price: Optional[float] = None
    ):
        def buy_internal():
            self.try_buy(code, amount, price)
            pos = self.position()
            pos_list = pos["data"]
            for info in pos_list:
                if info["证券代码"] == code:
                    return info

            return None

        ret = buy_internal()
        if ret is None:
            msg.send_no_except(f"try buy {code} failed")
        else:
            msg.send_no_except(
                f"buy {code} success.\n code name position: \n {ret["证券代码"]} {ret["证券名称"]} {ret["仓位占比(%)"]}"
            )

    def sell(
        self, code: str, amount: Optional[int] = None, price: Optional[float] = None
    ):
        def sell_internal():
            self.try_sell(code, amount, price)
            pos = self.position()
            pos_list = pos["data"]
            for info in pos_list:
                if info["证券代码"] == code:
                    return info
            return None

        ret = sell_internal()
        if ret is None:
            msg.send_no_except(f"try sell {code} success")
        else:
            msg.send_no_except(
                f"sell {code} failed.\n code name position: \n {ret["证券代码"]} {ret["证券名称"]} {ret["仓位占比(%)"]}"
            )

    def try_sell(
        self, code: str, amount: Optional[int] = None, price: Optional[float] = None
    ):
        if price and amount % 100 != 0:
            return {"status": -1, "data": "amount must be times of 100"}

        url = f"{self.server_url}/sell/?code={code}"

        if amount is not None:
            url += f"&amount={amount}"
        if price is not None:
            url += f"&price={price}"

        try:
            resp = requests.get(url)
        except Exception as e:
            return str(e)
        return json.loads(resp.text)

    def try_buy(
        self, code: str, amount: Optional[int] = None, price: Optional[float] = None
    ):
        if price and amount % 100 != 0:
            return {"status": -1, "data": "amount must be times of 100"}
        url = f"{self.server_url}/buy/?code={code}"

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

client = TraderClient(url="http://172.19.112.1:11122")

if __name__ == "__main__":
    # print(buy('510050',100))
    # print(sell('513060',100))
    # print(sell("513050"))
    client = TraderClient(url="http://172.19.112.1:11122")
    print(client.sell("159869"))
    # print(client.position())
    # try:
    #    print(user.buy('510880',None,None))
    # except Exception as e:
    #    print(e)
