import hashlib
import base64
import hmac
import requests
import time
import json

from pathlib import Path

# 获取当前脚本的父目录路径
current_dir = Path(__file__).parent


def gen_sign(timestamp, secret):
    # 拼接timestamp和secret
    string_to_sign = "{}\n{}".format(timestamp, secret)
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    # 对结果进行base64处理
    sign = base64.b64encode(hmac_code).decode("utf-8")
    return sign


def send(msg):
    config = json.load((current_dir / "secret.json").open("r"))
    timestamp = int(time.time())
    sign = gen_sign(timestamp, config["secret"])
    resp = requests.post(
        config["url"],
        json={
            "msg_type": "text",
            "content": {"text": f'<at user_id="all">所有人</at> {msg}'},
            "timestamp": str(timestamp),
            "sign": sign,
        },
    )
    return resp.json()


if __name__ == "__main__":
    print(send("交易测试"))
