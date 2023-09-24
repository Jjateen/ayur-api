from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:3000",
    # "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoringItem(BaseModel):
    Gender:int
    bodyBuild_Size:int
    bodyFrame_Breadth:int
    bodyFrame_Length:int
    bodyHair_Color:int
    chest_Breadth:int
    eye_Color:int
    eye_Size:int
    hair_Growth:int
    hair_Type:int
    nails_color:int
    palate_Color:int
    palms_Color:int
    scalpHair_Color:int
    skin_Color:int
    skin_Type:int
    soles_Color:int
    teeth_Color:int
    teeth_Shape:int
    teeth_Shape_Even:int
    appetite_Amount:int
    appetite_Frequency:int
    body_Odour:int
    bowel_Freq:int
    digestive_Amount:int
    like_Bitter:int
    like_Salty:int
    like_Sour:int
    like_Sweet:int
    sleep_Amount:int
    sleep_Quality:int
    stool_Consistency:int
    speaking_Speed:int
    walking_Amount:int
    walking_Speed:int
    healing_Power:int
    mental_Power:int
    physical_Power:int
    resistance_Power:int
    Anger_Quality:int
    forgetfulness_speed:int
    memorizing_speed:int
    hair_Nature3:int
    voice_clear:int
    skin_cracked:int
    skin_freckle:int
    skin_mark:int
    skin_mole:int
    skin_pimple:int
    skin_wrinkled:int

with open('clf.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    y_proba = model.predict_proba(df)
    y_proba *=100
    predicted_class = int(np.argmax(y_proba))
    confidence_values = y_proba.tolist()[0]
    return {"prediction": predicted_class, "confidence": confidence_values}

@app.get('/')
async def read_root():
    return "Welcome to Prakriti ChatBot!"
