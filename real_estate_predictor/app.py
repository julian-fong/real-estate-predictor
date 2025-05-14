from fastapi import FastAPI
from real_estate_predictor.predict import predict, extract_input

app = FastAPI()

# fastapi dev app.py
@app.get("/")
async def root():
    return {"msg": "please use the predict endpoint to generate predictions"}


@app.get("/predict/sale/{mlsNumber}")
async def predict_listing(mlsNumber, listing_type: str):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}
