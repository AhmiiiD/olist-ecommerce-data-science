from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# تحميل الموديل والـ encoders
with open('satisfaction_model.pkl', 'rb') as f:
    saved = pickle.load(f)
    
model = saved['model']
encoders = saved['encoders']
features = saved['features']

app = FastAPI(title="Olist Customer Satisfaction API")

# شكل البيانات اللي الـ API هيستقبلها
class OrderInput(BaseModel):
    order_value: float
    freight_value: float
    number_of_items: int
    payment_type: str
    payment_installments: int
    delivery_days: float
    delivery_delay: float
    is_late: int
    customer_order_count: int
    previous_orders: int
    customer_state: str
    product_category_name_english: str

@app.get("/")
def home():
    return {"message": "Olist Customer Satisfaction Prediction API"}

@app.post("/predict")
def predict(order: OrderInput):
    data = pd.DataFrame([order.dict()])
    
    # نطبق نفس الـ encoding اللي استخدمناه وقت التدريب
    for col in ['payment_type', 'customer_state', 'product_category_name_english']:
        le = encoders[col]
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')
        data[col] = data[col].apply(lambda x: x if x in le.classes_ else 'unknown')
        data[col] = le.transform(data[col])
    
    prediction = model.predict(data[features])[0]
    result = "Satisfied" if prediction == 1 else "Unsatisfied"
    
    return {"prediction": result}