import json
import re
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# ✅ 모델 로딩
model = BertForSequenceClassification.from_pretrained("./detect/kobert_offensive")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model.eval()

# ✅ 욕설 사전 로드
with open("data/badwords.json", "r", encoding="utf-8") as f:
    BADWORDS = set(json.load(f)["badwords"])

# ✅ 강제 차단 단어 리스트 로드
with open("data/force_block.json", "r", encoding="utf-8") as f:
    FORCE_BLOCK = set(json.load(f)["force_block"])

# ✅ 전처리 함수
def normalize(text):
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text).lower()

# ✅ 감지된 욕설 리스트 반환
def contains_badword(text):
    norm = normalize(text)
    detected = [word for word in BADWORDS if word in norm]
    if detected:
        print(f"✅ 감지된 욕설: {', '.join(detected)}")
    return detected

# ✅ KoBERT 예측 함수
def predict_kobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.argmax(logits, dim=1).item()

# ✅ 최종 욕설 감지 함수
def is_abuse(text):
    detected = contains_badword(text)

    if not detected:
        return False

    if any(word in FORCE_BLOCK for word in detected):
        return True

    return predict_kobert(text) == 1

# ✅ 테스트 실행
def main():
    print("⌨️ 테스트 문장을 입력하세요 (종료: q)")
    while True:
        text = input("입력: ")
        if text.lower() == "q":
            break

        if is_abuse(text):
            print("🚨 욕설 감지됨\n")
        else:
            print("✅ 정상 문장입니다\n")

if __name__ == "__main__":
    main()
