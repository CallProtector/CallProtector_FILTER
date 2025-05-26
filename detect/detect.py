import json
import re
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# ✅ 모델 및 토크나이저 로드
model = BertForSequenceClassification.from_pretrained("./detect/kobert_offensive")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model.eval()

# ✅ 욕설/강제 차단 단어 사전 로딩
with open("data/badwords.json", "r", encoding="utf-8") as f:
    BADWORDS = set(json.load(f)["badwords"])

with open("data/force_block.json", "r", encoding="utf-8") as f:
    FORCE_BLOCK = set(json.load(f)["force_block"])

# ✅ 전처리 함수
def normalize(text):
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text).lower()

# ✅ 단어 사전 기반 필터링
def contains_badword(text):
    norm = normalize(text)
    detected = [word for word in BADWORDS if word in norm]
    return detected

# ✅ KoBERT 기반 문맥 판단
def predict_kobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.argmax(logits, dim=1).item()

# ✅ 최종 판별
def is_abuse(text):
    detected = contains_badword(text)
    if not detected:
        return False, [], "정상"

    if any(word in FORCE_BLOCK for word in detected):
        return True, detected, "욕설(강제차단)"

    is_abuse = predict_kobert(text) == 1
    return is_abuse, detected, "욕설" if is_abuse else "정상"
