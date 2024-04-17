import torch
import torch.nn as nn
from model.bertclass import BERTSentenceTransform, BERTClassifier
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(device)

digital = 8
society = 9
economic = 9
culture = 11

config = dict(
    max_len=100,
    batch_size=64,
    warmup_ratio=0.1,
    num_epochs=3,
    max_grad_norm=1,
    log_interval=200,
    learning_rate=5e-5,
)

tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
bertmodel = BertModel.from_pretrained("skt/kobert-base-v1")
bertmodel.to(device)

one_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
hidden_size = 768

one_model.classifier = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size // 2, hidden_size // 4),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size // 4, 37),
)

one_model.load_state_dict(
    torch.load("./data/pt_file/all_37_model_9.pt", map_location=device)
)
# 0.6803024026512013

parent_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
parent_model.load_state_dict(
    torch.load("./data/pt_file/parent_model_9.pt", map_location=device)
)
# 0.8676470588235294

digital_model = BERTClassifier(bertmodel, num_classes=digital, dr_rate=0.5).to(device)
digital_model.load_state_dict(
    torch.load("./data/pt_file/digital_model_7.pt", map_location=device)
)
# 0.7420753588516746

society_model = BERTClassifier(bertmodel, num_classes=society, dr_rate=0.5).to(device)
society_model.load_state_dict(
    torch.load("./data/pt_file/society_model_9.pt", map_location=device)
)
# 0.7766262755102041

economic_model = BERTClassifier(bertmodel, num_classes=economic, dr_rate=0.5).to(device)
economic_model.load_state_dict(
    torch.load("./data/pt_file/economic_model_9.pt", map_location=device)
)
# 0.7933114035087719

culture_model = BERTClassifier(bertmodel, num_classes=culture, dr_rate=0.5).to(device)
culture_model.load_state_dict(
    torch.load("./data/pt_file/culture_model_6.pt", map_location=device)
)
# 0.796875

transform = BERTSentenceTransform(tokenizer, max_seq_length=config["max_len"], pad=True)
