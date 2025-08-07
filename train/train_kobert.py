import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, Trainer, TrainingArguments, AutoConfig
import torch.nn as nn
import torch

# ✅ 학습 설정
MODEL_NAME = "monologg/kobert"
NUM_CLASSES = 4 
THRESHOLD = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ✅ 데이터셋 로딩
def load_json_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ✅ 데이터 불러오기
verbal = load_json_dataset("./data/verbal_abuse_class.json")
sexual = load_json_dataset("./data/sexual_harassment_class.json")
threat = load_json_dataset("./data/threat_class.json")
normal = load_json_dataset("./data/normal_class.json")

# ✅ 전체 병합 및 셔플
full_data = verbal + sexual + threat + normal
np.random.shuffle(full_data)

# ✅ DataFrame 변환 및 split
df = pd.DataFrame(full_data)
train_df, valid_df = train_test_split(df[["text", "label"]], test_size=0.1, random_state=42, stratify=df["label"]) # Added stratify

# ✅ HuggingFace Dataset 변환
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

# ✅ 토크나이즈
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
valid_ds = valid_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
valid_ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

# ✅ 모델 정의
class KoBERTClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None): # Changed 'label' to 'labels'
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# ✅ 구성 및 초기화
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
model = KoBERTClassifier.from_pretrained(MODEL_NAME, config=config).to(device)

# ✅ 평가 지표
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# ✅ 학습 인자
training_args = TrainingArguments(
    output_dir="./model/kobert_v11",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)

# ✅ Trainer 정의 및 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
)

# ✅ 학습 시작
trainer.train()

# ✅ 모델 저장
trainer.save_model("./model/kobert_v9")
tokenizer.save_pretrained("./model/kobert_v9")