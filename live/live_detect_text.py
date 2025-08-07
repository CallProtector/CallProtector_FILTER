import torch
import re
import json
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel

# ✅ 설정
LABEL_NAMES = ["욕설", "성희롱", "협박", "정상"]
MODEL_PATH = "./model/kobert_v11"
THRESHOLD = 0.9
DELTA_THRESHOLD = 0.2
NORMAL_CLASS_INDEX = 3

# ✅ 단어 사전 로딩
with open("data/badwords.json", encoding="utf-8") as f:
    BADWORDS = set(json.load(f)["badwords"])

with open("data/force_block.json", encoding="utf-8") as f:
    FORCE_BLOCK = set(json.load(f)["force_block"])

# ✅ KoBERT 모델 정의
class KoBERTClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits

# ✅ 전처리 함수
def normalize(text: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text).lower().strip()

def contains_badword(norm_text: str):
    return [word for word in BADWORDS if word in norm_text]

# ✅ 모델 및 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH)
model = KoBERTClassifier.from_pretrained(MODEL_PATH, config=config)
model.eval()

# ✅ 예측 함수
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs)
        probs = F.softmax(logits, dim=1)[0].tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = LABEL_NAMES[pred_idx]
    return probs, pred_label, pred_idx

# ✅ 최종 판별 함수
def is_abuse(text: str):
    norm = normalize(text)
    detected = contains_badword(norm)

    # 강제 차단 단어 감지
    if any(word in FORCE_BLOCK for word in detected):
        return True, detected, "욕설(강제차단)", [1.0, 0.0, 0.0, 0.0]

    probs, pred_label, pred_idx = predict(text)
    max_prob = max(probs[:NORMAL_CLASS_INDEX])
    normal_prob = probs[NORMAL_CLASS_INDEX]
    delta = max_prob - normal_prob

    is_abusive = (
        pred_idx != NORMAL_CLASS_INDEX and
        max_prob > THRESHOLD and
        delta > DELTA_THRESHOLD and
        normal_prob < 0.3
    )

    # 오탐 방지 로직
    if is_abusive and not detected:
        if max_prob < 0.97 or normal_prob > 0.2:
            is_abusive = False

    return is_abusive, detected, pred_label, probs

# ✅ 테스트
if __name__ == "__main__":
    print("📨 테스트할 문장을 입력하세요 (종료: 'exit'):")
    while True:
        try:
            text = input(">>> ")
            if text.lower().strip() == "exit":
                break

            norm = normalize(text)
            print(f"🔍 정규화된 텍스트: '{norm}'")

            is_abusive, detected, label, probs = is_abuse(text)

            if label == "욕설(강제차단)":
                print("🚨 판별 결과: 욕설(강제차단) 감지됨")
            else:
                print(f"📈 예측 확률: {probs}")
                if is_abusive:
                    print(f"🚨 판별 결과: {label} 감지됨")
                else:
                    print("✅ 판별 결과: 정상 문장입니다.")

            if detected:
                print(f"🙊 감지된 욕설 단어: {', '.join(detected)}")

            print()
        except Exception as e:
            print(f"❗ 오류 발생: {e}")
