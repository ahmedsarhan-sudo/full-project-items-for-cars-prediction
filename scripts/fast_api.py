
from fastapi import FastAPI,Query ,Path ,Body , HTTPException
from pydantic import BaseModel , Field
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles


app = FastAPI()



BASE_DIR = Path(__file__).resolve().parent.parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR),
    name="static"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def prediction_page():
    return FileResponse(BASE_DIR / "scripts" / "prediction.html")


with open(r'../models/one_hot_encoder.pkl' , 'rb') as f:
    onehot_encoding = pickle.load(f)

with open(r'../models/label_encoders.pkl' ,'rb') as f:
    label_encoded = pickle.load(f)

with open(r'../models/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

with open(r'../models/model.pkl', 'rb') as f:
    Random_Forest = pickle.load(f)

with open("../models/manufacturer_model_map.pkl", "rb") as f:
    manufacturer_model_map = pickle.load(f)

class Inputs_Car(BaseModel):
    Levy: int = Field(...,ge=0 , title='Levy')
    Manufacturer: str = Field(..., min_length=2 , max_length = 35 , title="Manufacturer")
    Model: str = Field(..., min_length=2,max_length = 35 , title='model')
    Prod_year: int = Field(..., title='prod year')
    Category: str = Field(..., min_length=2 , max_length = 25 , title="Category")
    Leather_interior: str = Field(..., min_length=2 , max_length = 3 , title="lenther")
    Fuel_type: str = Field(..., min_length=2 , max_length = 25 , title="Category")
    Engine_volume: float = Field(...,ge=0 , title='engine')
    Mileage: int = Field(...,ge=0 , title='milage')
    Cylinders: float = Field(...,ge=0 , title='cyliners')
    Gear_box_type: str = Field(..., min_length=2 , max_length = 25 , title="gare box type")
    Drive_wheels: str = Field(..., min_length=2 , max_length = 25 , title="drive wheeles")
    Wheel: str = Field(..., min_length=2 , max_length = 25 , title="wheele")
    Color: str = Field(..., min_length=2 , max_length = 25 , title="color")
    Airbags: int = Field(...,ge=0 , title='milage')

@app.post('/predict')
async def add_feature(car:Inputs_Car):

    try:

        data = pd.DataFrame([car.dict()])
        data['Age'] = datetime.now().year - data['Prod_year']
        data = data.drop(columns=['Doors', 'Prod_year'], errors='ignore')

        column_rename_map = {
            "Leather_interior": "Leather interior",
            "Gear_box_type": "Gear box type",
            "Drive_wheels": "Drive wheels",
            "Engine_volume": "Engine volume",
            "Fuel_type": "Fuel type"
        }
        data.rename(columns=column_rename_map, inplace=True)

        columns_one_hot_encoding =['Gear box type' , 'Drive wheels' , 'Wheel' ,'Fuel type']

        encoded_data = onehot_encoding.transform(data[columns_one_hot_encoding])
        encoded_data_df = pd.DataFrame(encoded_data , columns=onehot_encoding.get_feature_names_out(columns_one_hot_encoding) , index = data.index)
        data = pd.concat([data,encoded_data_df] , axis = 1)
        data = data.drop(columns = columns_one_hot_encoding)

        columns_label_encoding = ['Manufacturer' , 'Model' , 'Category' ,'Color']

        for col in columns_label_encoding:
            le = label_encoded[col]
            data[col] = le.transform(data[col])

        numerical_columns = ['Levy','Engine volume', 'Mileage' , 'Age']

        data[numerical_columns]  = scaler.transform(data[numerical_columns])
        data['Leather interior'] = data['Leather interior'].map({'Yes' : 1 , 'No':0})

        prediction = Random_Forest.predict(data)

        return {'Prediction' : prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400 , detail=str(e))
    

@app.get("/manufacturers")
def get_manufacturers():
    return {"manufacturers": label_encoded["Manufacturer"].classes_.tolist()}


@app.get("/models/{manufacturer}")
def get_models(manufacturer: str):
    try:
        models = manufacturer_model_map.get(manufacturer, [])
        return {"models": models}
    except:
        return {"models": []}
    