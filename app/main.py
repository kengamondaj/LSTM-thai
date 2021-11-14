from fastapi import FastAPI
from pydantic import BaseModel

from app.my_sentiment_model import MySentimentModel

class Input(BaseModel):
    text: str

app = FastAPI()

@app.post("/")
async def predicts(item: Input):
    model = MySentimentModel
    res = model.predict(item.text)
    return {'sentiment':res}

@app.get("/")
def read_root():
    return {"Hello": "World"}
