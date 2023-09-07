from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from core import *


app = FastAPI()

@app.post("/inputchat")
async def generate_response(question: str):
    candidates, output = get_candidates(question)
    response = get_response(output, candidates)
    return {"answer": response}
