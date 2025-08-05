from fastapi import FastAPI
from pydantic import BaseModel
from detect.detect import is_abuse

app = FastAPI()

class AbuseRequest(BaseModel):
    text: str

@app.post("/api/abuse/filter")
def filter_abuse(req: AbuseRequest):
    abuse, detected, abuse_type = is_abuse(req.text)
    return {
        "abuse": abuse,
        "detected": bool(detected),
        "type": abuse_type
    }
