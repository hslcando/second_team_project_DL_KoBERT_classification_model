import json
import requests
import gradio as gr

# 아래는 SESAC OPEN1 IPv4 주소
BACKEND_URL = "http://172.16.219.143:8000"  # IPv4 주소 입력 > cmd창에서 ipconfig로 확인 


def news_classifier(text, history):
    payload = {"msg": text}
    response = requests.post(
        BACKEND_URL + "/news_class", data=json.dumps(payload)
    ).json()
    news_class = response["result"]
    probs = response["probs"]
    return "\n\n".join((probs, news_class))


demo = gr.ChatInterface(
    news_classifier, description="분류하고 싶은 기사를 입력해주세요!"
)
demo.launch(share=True)
