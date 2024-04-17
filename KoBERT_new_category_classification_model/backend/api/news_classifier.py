from fastapi import APIRouter
from schema.request import News
import re

import torch
import torch.nn.functional as F
import numpy as np

from model.models import one_model, transform

from model.models import parent_model, transform
from model.models import digital_model, society_model, economic_model, culture_model
from data.topic_list import labels

router = APIRouter()

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(device)

idx2label = {0: "digital", 1: "society", 2: "economic", 3: "culture"}
sub_models = [digital_model, society_model, economic_model, culture_model]


def predict(predict_sentence):
    token_ids, valid_length, segment_ids = transform(predict_sentence)
    token_ids = torch.tensor([token_ids]).to(device)
    segment_ids = torch.tensor(segment_ids).to(device)

    parent_model.eval()
    out = parent_model(token_ids, [valid_length], segment_ids)

    logits = out[0].detach().cpu().numpy()
    idx = np.argmax(logits)

    topic_name = idx2label[idx]
    sub_model = sub_models[idx]
    sub_topics = labels[topic_name]
    label_num = len(sub_topics)

    sub_model.eval()
    out = sub_model(token_ids, [valid_length], segment_ids)

    prob = F.softmax(out, dim=1)[0]
    probs = []
    for i in range(label_num):
        probs.append(f"{sub_topics[i]}: {prob[i] * 100:.2f}%")
    print("\n".join(probs))
    logits = out[0].detach().cpu().numpy()
    return (
        f">> 입력하신 기사는 {topic_name}의 {sub_topics[np.argmax(logits)]} 기사입니다.",
        "\n".join(probs),
    )


topics = {}
for (topic, sub_topics), i in zip(labels.items(), (0, 8, 17, 26)):
    topic_dict = {i: [topic, sub_topic] for i, sub_topic in enumerate(sub_topics, i)}
    topics.update(topic_dict)


# def predict0(predict_sentence):
#     token_ids, valid_length, segment_ids = transform(predict_sentence)
#     token_ids = torch.tensor([token_ids]).to(device)
#     segment_ids = torch.tensor(segment_ids).to(device)

#     one_model.eval()
#     out = one_model(token_ids, [valid_length], segment_ids)

#     prob = F.softmax(out, dim=1)[0]
#     probs = []
#     for i in range(37):
#         probs.append(f"{topics[i][0]}_{topics[i][1]}: {prob[i] * 100:.2f}%")
#     print("\n".join(probs))
#     logits = out[0].detach().cpu().numpy()
#     idx = np.argmax(logits)
#     return (
#         f">> 입력하신 기사는 {topics[idx][0]}의 {topics[idx][1]} 기사입니다.",
#         "\n".join(probs),
#     )


@router.post("/news_class")
def classifier(request: News):
    text = request.msg
    text = re.sub(r"[^ 가-힣]", "", text)
    text = text = re.sub(r" +", " ", text)

    result, probs = predict(text.strip())
    # result, probs = predict0(text.strip())
    return {"result": result, "probs": probs}
