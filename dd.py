import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
import json


# url = f'https://oapi.dingtalk.com/robot/send?access_token=4a67c5851e9348fa50b18d4cf9a2a606c1faa23c7775d7526ca4d08df8108220&timestamp={timestamp}&sign={sign}'
#  -H 'Content-Type: application/json' \
#  -d '{"msgtype": "text","text": {"content":"我就是我, 是不一样的烟火"}}'
# print(url)

# def test_dingding_info(message, webhook):
#     data={"msgtype":"text","text":{"content": message}}
#     requests.post(webhook, json=data)

# res = requests.post(url,json={"msgtype": "text","text": {"content":"我就是我, 是不一样的烟火"}})
# print(res.text)


class Msg:
    url = "https://oapi.dingtalk.com/robot/send?access_token=4a67c5851e9348fa50b18d4cf9a2a606c1faa23c7775d7526ca4d08df8108220"

    secret = "SEC01d23346b4a79697893e0770a42438d662dec3d18aaf45b4321626dd7d714e6a"

    def send(text):
        timestamp = str(round(time.time() * 1000))
        secret_enc = Msg.secret.encode("utf-8")
        string_to_sign = "{}\n{}".format(timestamp, Msg.secret)
        string_to_sign_enc = string_to_sign.encode("utf-8")
        hmac_code = hmac.new(
            secret_enc, string_to_sign_enc, digestmod=hashlib.sha256
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        url = Msg.url
        url += f"&timestamp={timestamp}&sign={sign}"

        res = requests.post(
            url, json={"msgtype": "text", "text": {"content": f"{text}"}}
        )
        return json.loads(res.text)


if __name__ == "__main__":
    # msg = Msg()
    Msg.send("hello")
