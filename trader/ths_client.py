import requests
import json

URL_BASE = "http://127.0.0.1:11122"

def balance():
    resp = requests.get(f"{URL_BASE}/balance")

    return json.loads(resp.text)

def position():
    resp = requests.get(f"{URL_BASE}/position")

    return json.loads(resp.text)

def try_buy(code):
    url = f"{URL_BASE}/buy/?code={code}"

    resp = requests.get(url)

    return json.loads(resp.text)

def buy(code):


if __name__ == "__main__":

    # print(try_buy("002205",None,None))
    # print(position())
    print(balance())




# docker network create -d macvlan \
#     --subnet=20.20.0.0/24 \
#     --gateway=20.20.0.1 \
#     --ip-range=192.168.0.96/28 \
#     -o parent=eth0 vlan
