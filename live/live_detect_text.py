import json
import re
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# ✅ 모델 로딩
model = BertForSequenceClassification.from_pretrained("./model/kobert_multi_all")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model.eval()

# ✅ 욕설 사전 및 강제 차단 단어 로딩
with open("data/badwords.json", "r", encoding="utf-8") as f:
    BADWORDS = set(json.load(f)["badwords"])

with open("data/force_block.json", "r", encoding="utf-8") as f:
    FORCE_BLOCK = set(json.load(f)["force_block"])

# ✅ 라벨 인덱스 → 이름
LABELS = ["욕설", "성희롱", "협박"]

# ✅ 전처리 함수
def normalize(text):
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text).lower()

# ✅ 단어 사전 기반 감지
def contains_badword(text):
    norm = normalize(text)
    detected = [word for word in BADWORDS if word in norm]
    if detected:
        print(f"📌 감지된 욕설 단어: {', '.join(detected)}")
    return detected

# ✅ 다중 라벨 예측 함수
def predict_kobert_multi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        print(f"📊 예측 확률: {probs}")
        return [LABELS[i] for i, p in enumerate(probs) if p >= 0.4]

# ✅ 최종 감지 함수
def is_abuse(text):
    detected = contains_badword(text)

    # 강제 차단 우선
    if any(word in FORCE_BLOCK for word in detected):
        return True, detected, ["욕설(강제차단)"]

    # 다중 라벨 감지
    abuse_types = predict_kobert_multi(text)

    return bool(abuse_types or detected), detected, abuse_types if abuse_types else ["정상"]

# ✅ 콘솔 테스트
def main():
    print("⌨️ 테스트 문장을 입력하세요 (종료: q)")
    while True:
        text = input("입력: ")
        if text.lower() == "q":
            break

        abuse, detected, abuse_types = is_abuse(text)

        if abuse:
            print(f"🚨 감지됨 → 유형: {', '.join(abuse_types)}\n")
        else:
            print("✅ 정상 문장입니다\n")

if __name__ == "__main__":
    main()
