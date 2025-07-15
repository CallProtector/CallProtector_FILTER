FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 소스 복사
COPY . .


# 포트 오픈 (필요 시, FastAPI라면 8000)
EXPOSE 8000

# 앱 실행 명령어 (FastAPI 기준 예시)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
