# speech_recognition 사용하여 마이크 임시 테스트
import speech_recognition as sr
import torch
from pathlib import Path

from transformers import BertForSequenceClassification, AutoTokenizer

model = BertForSequenceClassification.from_pretrained("./detect/kobert_offensive")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.argmax(logits, dim=1).item()

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    print("🎤 마이크에서 음성을 듣고 있습니다. (Ctrl+C로 종료)")

    while True:
        try:
            with mic as source:
                print("🗣 음성을 입력하세요...")
                audio = recognizer.listen(source, timeout=5)

            text = recognizer.recognize_google(audio, language='ko-KR')
            print(f"📝 텍스트 변환: {text}")

            result = predict(text)
            if result == 1:
                print("🚨 악플/욕설 감지됨")
            else:
                print("✅ 정상")

        except KeyboardInterrupt:
            print("\n🛑 종료")
            break
        except Exception as e:
            print(f"⚠️ 에러 발생: {e}")

if __name__ == "__main__":
    main()
