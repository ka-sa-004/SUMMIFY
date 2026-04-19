from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

@app.get("/")
def home():
    return {"message": "Summify API running"}

@app.post("/summarize")
def summarize(data: dict):
    text = data.get("text")

    response = requests.post(API_URL, json={"inputs": text})
    result = response.json()

    try:
        return {"summary": result[0]["summary_text"]}
    except:
        return {"summary": "Error generating summary"}