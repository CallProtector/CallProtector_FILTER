from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import concatenate_datasets
import torch.nn as nn
import numpy as np
import pandas as pd
import json

# ✅ 욕설/성희롱 데이터 로드
dataset = load_dataset("smilegate-ai/kor_unsmile")
target_labels = ["악플/욕설", "여성/가족", "협박"]

def convert_labels(example):
    return {
        "text": example["문장"],
        "labels": [
            int(example["악플/욕설"]),
            int(example["여성/가족"]),
            0
        ]
    }

train_data = dataset["train"].map(convert_labels)
valid_data = dataset["valid"].map(convert_labels)

# ✅ 협박 데이터 로드
with open("data/threat_sentences.json", "r", encoding="utf-8") as f:
    threat_samples = json.load(f)

threat_dataset = Dataset.from_pandas(pd.DataFrame(threat_samples))

# ✅ 협박 문장을 학습 데이터에 병합
train_data = concatenate_datasets([train_data, threat_dataset])

# ✅ KoBERT 토크나이저 로드 및 적용
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

train_data = train_data.map(tokenize, batched=True)
valid_data = valid_data.map(tokenize, batched=True)

# 라벨 float 변환
def cast_labels(example):
    example["labels"] = [float(x) for x in example["labels"]]
    return example

train_data = train_data.map(cast_labels)
valid_data = valid_data.map(cast_labels)

# cast_column으로 라벨 float32 확정
from datasets import Value, Sequence
train_data = train_data.cast_column("labels", Sequence(Value("float32")))
valid_data = valid_data.cast_column("labels", Sequence(Value("float32")))

# 최종 torch 포맷
train_data.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
valid_data.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])


# ✅ KoBERT 모델 정의 (다중 라벨 분류용)
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3)
model.config.problem_type = "multi_label_classification"

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./model/kobert_multi_all",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_total_limit=1,
    logging_dir="./logs",
)

# ✅ 평가 지표 정의
def compute_metrics(pred):
    probs = 1 / (1 + np.exp(-pred.predictions))
    preds = (probs >= 0.5).astype(int)
    labels = pred.label_ids
    acc = np.mean((preds == labels).astype(float))
    return {"accuracy": acc}

# ✅ Trainer 정의 및 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    compute_metrics=compute_metrics
)

trainer.train()

# ✅ 학습된 모델과 토크나이저 저장
trainer.save_model("./model/kobert_multi_all")
tokenizer.save_pretrained("./model/kobert_multi_all")
