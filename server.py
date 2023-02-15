from fastapi import FastAPI
from app.model.prediction import load_model, predict
from pydantic import BaseModel

app = FastAPI();

class Prediction_Out(BaseModel):
    prediction = str

class TextIn(BaseModel):
    type_of = str
    text = str

@app.get('/')
def hello_world():
    return {"hello":"world"}

@app.post('/text-generate',response_model=Prediction_Out)
def predict(payload:TextIn):
    load_model()
    prediction = predict(dict(payload))
    return {"predicted result":prediction}