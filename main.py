from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import uvicorn

import tensorflow as tf
import numpy as np
from transformer import summarize  # custom functions

# FastAPI setup
app = FastAPI()
# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Input data model
class TextInput(BaseModel):
    original_text: str

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Summarization API endpoint
@app.post("/summarize")
async def summarize_text_api(input: TextInput):
    summary = summarize(input.original_text)  # Call your prediction function
    return {"summary": summary}

# Run with: uvicorn main:app --reload
# http://127.0.0.1:8000
# start chrome http://127.0.0.1:8000

