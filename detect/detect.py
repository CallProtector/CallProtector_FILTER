import json
import re
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# ✅ 모델 및 토크나이저 로드
#model = BertForSequenceClassification.from_pretrained("./model/kobert_multi_all")
# 도커로 배포를 위해 수정
model = BertForSequenceClassification.from_pretrained("./detect/kobert_offensive")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model.eval()

# ✅ 단어 사전 로딩
with open("data/badwords.json", "r", encoding="utf-8") as f:
    BADWORDS = set(json.load(f)["badwords"])

with open("data/force_block.json", "r", encoding="utf-8") as f:
    FORCE_BLOCK = set(json.load(f)["force_block"])

# ✅ 전처리
def normalize(text):
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text).lower().strip()

# ✅ 단어 사전 기반 필터링
def contains_badword(text):
    norm = normalize(text)
    print(f"🔍 정규화된 텍스트: '{norm}'")
    detected = [word for word in BADWORDS if word in norm]
    if detected:
        print(f"📌 감지된 욕설 단어: {', '.join(detected)}")
    return detected

# ✅ 다중 라벨 KoBERT 예측
def predict_kobert_multi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().tolist()  # [욕설, 성희롱, 협박]
        return [i for i, p in enumerate(probs) if p >= 0.5]

# ✅ 라벨 인덱스 → 이름 매핑
LABEL_NAMES = ["욕설", "성희롱", "협박"]

# ✅ 최종 판별 함수
def is_abuse(text):
    detected = contains_badword(text)

    if any(word in FORCE_BLOCK for word in detected):
        return True, detected, "욕설(강제차단)"

    abuse_indices = predict_kobert_multi(text)

    if not abuse_indices and not detected:
        return False, [], "정상"

    abuse_types = [LABEL_NAMES[i] for i in abuse_indices]
    return bool(abuse_types), detected, ", ".join(abuse_types) if abuse_types else "정상"
