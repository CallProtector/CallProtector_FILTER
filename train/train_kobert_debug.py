
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import torch
import json
from datasets import Value, Sequence

# ✅ 1. 욕설/성희롱 기본 데이터 로드
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

train_data = dataset["train"].select(range(300)).map(convert_labels)

# ✅ 2. 협박 데이터 로드
with open("data/threat_sentences.json", "r", encoding="utf-8") as f:
    threat_samples = json.load(f)
threat_dataset = Dataset.from_pandas(pd.DataFrame(threat_samples))

# ✅ 3. 성희롱 강화 데이터 로드
with open("data/sexualHarass_sentences.json", "r", encoding="utf-8") as f:
    sexism_samples = json.load(f)
sexism_dataset = Dataset.from_pandas(pd.DataFrame(sexism_samples))

# ✅ 4. 병합
train_data = concatenate_datasets([train_data, threat_dataset, sexism_dataset])

# ✅ 5. 토크나이저 및 전처리
tokenizer = BertTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

train_data = train_data.map(tokenize)

# ✅ 6. 라벨 float 변환 및 cast
def cast_labels(example):
    example["labels"] = [float(x) for x in example["labels"]]
    return example

train_data = train_data.map(cast_labels)
train_data = train_data.cast_column("labels", Sequence(Value("float32")))
train_data.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# ✅ 7. KoBERT 모델 정의
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3)
model.config.problem_type = "multi_label_classification"

# ✅ 8. 학습 설정
training_args = TrainingArguments(
    output_dir="./model/kobert_full_boost",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    logging_dir="./logs",
    logging_steps=5,
    save_total_limit=1,
    report_to="none"
)

# ✅ 9. 평가 지표 정의
def compute_metrics(pred):
    probs = 1 / (1 + np.exp(-pred.predictions))
    preds = (probs >= 0.4).astype(int)
    labels = pred.label_ids
    acc = np.mean((preds == labels).astype(float))
    return {"accuracy": acc}

# ✅ 10. Trainer 정의 및 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics
)

trainer.train()

# ✅ 11. 모델 저장
trainer.save_model("./model/kobert_debug")
tokenizer.save_pretrained("./model/kobert_debug")
