from fastapi import FastAPI
from pydantic import BaseModel

from pipeline.run import process

app = FastAPI()


class ProcessRequest(BaseModel):
    text: str


@app.post("/process")
def process_text(req: ProcessRequest) -> dict:
    return process(req.text)
