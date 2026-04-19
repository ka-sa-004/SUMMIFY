from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔹 Load model
MODEL_PATH = "./saved_summary_model"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# 🔹 device (FIXED)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():   # ❌ typo fixed
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()  # 🔥 IMPORTANT



class DialogueInput(BaseModel):
    dialogue: str

# 🔹 clean function (safe)
def clean_data(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# 🔥 MAIN FUNCTION
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    # ✅ IMPORTANT: T5 needs prefix
    input_text = "summarize: " + dialogue

    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # no_grad = faster inference
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# 🔹 API
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


# 🔹 UI
@app.get("/")
async def home():
    return FileResponse("index.html")