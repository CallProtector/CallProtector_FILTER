# train/train_kobert.py
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# 1. 데이터 불러오기
dataset = load_dataset("smilegate-ai/kor_unsmile")
train_data = dataset["train"].remove_columns([col for col in dataset["train"].column_names if col not in ["문장", "악플/욕설"]])
valid_data = dataset["valid"].remove_columns([col for col in dataset["valid"].column_names if col not in ["문장", "악플/욕설"]])
train_data = train_data.rename_column("문장", "text").rename_column("악플/욕설", "label")
valid_data = valid_data.rename_column("문장", "text").rename_column("악플/욕설", "label")

# 2. 토크나이저
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
train_data = train_data.map(tokenize, batched=True)
valid_data = valid_data.map(tokenize, batched=True)

# 3. 모델 정의
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)

# 4. 학습 설정
training_args = TrainingArguments(
    output_dir="./model/kobert_offensive",
    evaluation_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    save_total_limit=1,
    logging_dir="./logs",
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data
)

trainer.train()
trainer.save_model("./model/kobert_offensive")
tokenizer.save_pretrained("./model/kobert_offensive")

# tokenizer.save_pretrained("./model/kobert_offensive", legacy_format=True)