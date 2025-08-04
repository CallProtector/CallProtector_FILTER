import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    AutoTokenizer,
    BertPreTrainedModel,
    BertModel,
    Trainer,
    TrainingArguments,
    AutoConfig
)

# ✅ 하이퍼파라미터
THRESHOLD = 0.4
NUM_LABELS = 3

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# ✅ 데이터셋 로딩
def load_json_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

abuse_data = load_json_dataset("./data/verbal_abuse_dataset.json")
sexual_data = load_json_dataset("./data/sexual_harassment_dataset.json")
threat_data = load_json_dataset("./data/threat_dataset.json")
normal_data = load_json_dataset("./data/normal_dataset.json")

full_data = abuse_data + sexual_data + threat_data + normal_data
np.random.shuffle(full_data)

# ✅ train/valid split
train_raw, valid_raw = train_test_split(full_data, test_size=0.1, random_state=42)

# ✅ HuggingFace Dataset 변환
features = Features({
    "text": Value("string"),
    "labels": Sequence(Value("float32"), length=NUM_LABELS)
})
train_ds = Dataset.from_pandas(pd.DataFrame(train_raw), features=features)
valid_ds = Dataset.from_pandas(pd.DataFrame(valid_raw), features=features)

# ✅ 토크나이즈 + 라벨 전처리
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
valid_ds = valid_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
valid_ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# ✅ 모델 정의
class KoBERTForMultiLabel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # ✅ pos_weight 적용 (가중치 = 총 데이터 수 / 해당 클래스 수)
        total = 4200
        weights = torch.tensor([
            total / 1000,  # 욕설
            total / 1000,  # 성희롱
            total / 1000   # 협박
        ]).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# ✅ 모델 설정 및 초기화
config = AutoConfig.from_pretrained("monologg/kobert", num_labels=NUM_LABELS, problem_type="multi_label_classification")
model = KoBERTForMultiLabel.from_pretrained("monologg/kobert", config=config).to(device)

# ✅ 평가 지표
def compute_metrics(pred):
    probs = 1 / (1 + np.exp(-pred.predictions))
    preds = (probs >= THRESHOLD).astype(int)
    labels = pred.label_ids
    f1s = f1_score(labels, preds, average=None)
    print(f"📊 F1 per label → 욕설: {f1s[0]:.4f}, 성희롱: {f1s[1]:.4f}, 협박: {f1s[2]:.4f}")
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# ✅ 학습 인자
training_args = TrainingArguments(
    output_dir="./model/kobert_abuse_classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    eval_steps=500,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir="./logs",
)

# ✅ Trainer 정의 및 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ✅ 모델 및 토크나이저 저장
save_dir = "./model/kobert_abuse_classifier"
trainer.save_model(save_dir)
