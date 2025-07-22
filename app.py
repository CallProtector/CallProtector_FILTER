from fastapi import FastAPI
from pydantic import BaseModel
from detect.detect import is_abuse

app = FastAPI() # ✅ springboot 경로 충돌을 방지하기 위해 제거

class AbuseRequest(BaseModel):
    text: str

@app.post("/filter-abuse")
def filter_abuse(req: AbuseRequest):
    abuse, detected, abuse_type = is_abuse(req.text)
    return {
        "abuse": abuse,
        "detected": bool(detected),
        "type": abuse_type
    }
