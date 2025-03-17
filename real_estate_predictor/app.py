from fastapi import FastAPI
from real_estate_predictor.predict import predict, extract_input
app = FastAPI()

#fastapi dev app.py
@app.get("/")
async def root():
    return {"msg": "please use the predict endpoint to generate predictions"}

@app.get("/predict/{mlsNumber}")
async def predict_listing(mlsNumber):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}

