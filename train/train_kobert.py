import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from datasets import load_dataset, Dataset, concatenate_datasets, Value, Sequence
from transformers import (
    AutoTokenizer,
    BertPreTrainedModel,
    BertModel,
    Trainer,
    TrainingArguments,
    AutoConfig
)

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ KoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# ✅ Unsmile 로드 및 변환
dataset = load_dataset("smilegate-ai/kor_unsmile")

def convert_labels(example):
    return {
        "text": example["문장"],
        "labels": [
            int(example["악플/욕설"]),
            int(example["여성/가족"]),
            0
        ]
    }

# ✅ Unsmile 일부 샘플 사용
train_data = dataset["train"].map(convert_labels).shuffle(seed=42).select(range(3000))
valid_data = dataset["valid"].map(convert_labels).shuffle(seed=42).select(range(500))

# ✅ 욕설 라벨만 따로 추출해서 추가
abusive_only = [ex for ex in dataset["train"] if ex["악플/욕설"] == 1]
abusive_df = pd.DataFrame([{"text": ex["문장"], "labels": [1, 0, 0]} for ex in abusive_only])
abusive_ds = Dataset.from_pandas(abusive_df)
train_data = concatenate_datasets([train_data, abusive_ds])

# ✅ 성희롱/협박 강화 데이터 로드
with open("data/threat_sentences.json", encoding="utf-8") as f:
    threat_samples = json.load(f)
with open("data/sexualHarass_sentences.json", encoding="utf-8") as f:
    sexual_samples = json.load(f)

# ✅ 성희롱/협박 데이터 증강
augmented_threat = []
for item in threat_samples:
    txt = item["text"]
    labels = item["labels"]
    variants = list(set([
        txt,
        txt.replace("너", "니"),
        txt.replace("너", "너 그거 알아?"),
        txt.replace("죽여", "끝장내"),
        txt.replace("패버", "부셔버"),
        txt.replace("죽고싶", "살고싶지 않"),
        txt.replace("끝났", "끝장났"),
        txt + " 각오해",
    ]))
    for variant in variants:
        augmented_threat.append({"text": variant, "labels": labels})

augmented_sexual = []
for item in sexual_samples:
    txt = item["text"]
    labels = item["labels"]
    variants = list(set([
        txt,
        txt.replace("야", "이봐"),
        "근데 " + txt,
        txt.replace("진짜", "완전"),
        txt.replace("좋다", "끝내준다"),
    ]))
    for variant in variants:
        augmented_sexual.append({"text": variant, "labels": labels})

# ✅ 성희롱/협박 데이터 분할 및 결합
extra_samples = augmented_threat + augmented_sexual
train_extra, val_extra = train_test_split(extra_samples, test_size=0.1, random_state=42)
train_extra_ds = Dataset.from_pandas(pd.DataFrame(train_extra))
val_extra_ds = Dataset.from_pandas(pd.DataFrame(val_extra))

train_data = concatenate_datasets([train_data, train_extra_ds])
valid_data = concatenate_datasets([valid_data, val_extra_ds])

# ✅ 전처리 및 라벨 float 변환
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

def cast_labels(example):
    example["labels"] = [float(x) for x in example["labels"]]
    return example

train_data = train_data.map(tokenize, batched=True).map(cast_labels)
valid_data = valid_data.map(tokenize, batched=True).map(cast_labels)

train_data = train_data.cast_column("labels", Sequence(Value("float32")))
valid_data = valid_data.cast_column("labels", Sequence(Value("float32")))

train_data.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
valid_data.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# ✅ KoBERT 모델 정의 (가중치 적용)
class WeightedKoBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0, 2.5, 2.0]).to(device))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

config = AutoConfig.from_pretrained("monologg/kobert", num_labels=3, problem_type="multi_label_classification")
model = WeightedKoBERT.from_pretrained("monologg/kobert", config=config).to(device)

# ✅ 평가 지표
def compute_metrics(pred):
    probs = 1 / (1 + np.exp(-pred.predictions))
    preds = (probs >= 0.4).astype(int)
    labels = pred.label_ids
    f1s = f1_score(labels, preds, average=None)
    print(f"📊 F1 per label → 욕설: {f1s[0]:.4f}, 성희롱: {f1s[1]:.4f}, 협박: {f1s[2]:.4f}")
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./model/kobert_v4",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    compute_metrics=compute_metrics
)

trainer.train()

# ✅ 저장
save_dir = "./model/kobert_v4"
trainer.save_model(save_dir)
tokenizer.save_vocabulary(save_dir)
with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer.init_kwargs, f, ensure_ascii=False, indent=2)
