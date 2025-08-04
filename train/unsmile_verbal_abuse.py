from datasets import load_dataset, concatenate_datasets
import json

# ✅ Unsmile 데이터셋 로드
dataset = load_dataset("smilegate-ai/kor_unsmile")

# ✅ '악플/욕설' 라벨이 1인 문장만 필터링
train_abuse = dataset["train"].filter(lambda x: x["악플/욕설"] == 1)
valid_abuse = dataset["valid"].filter(lambda x: x["악플/욕설"] == 1)

# ✅ train + valid 합치기 (오류 수정)
abuse_all = concatenate_datasets([train_abuse, valid_abuse]).shuffle(seed=42)

# ✅ 상위 1000개 추출
abuse_1000 = abuse_all.select(range(1000))

# ✅ 'text'와 'labels'만 포함하는 형태로 변환
abuse_1000 = abuse_1000.map(lambda x: {
    "text": x["문장"],
    "labels": [1, 0, 0]
})

# ✅ 리스트로 변환 후 JSON 저장
abuse_cleaned_list = [{"text": item["text"], "labels": item["labels"]} for item in abuse_1000]

with open("./data/verbal_abuse_dataset.json", "w", encoding="utf-8") as f:
    json.dump(abuse_cleaned_list, f, ensure_ascii=False, indent=2)
