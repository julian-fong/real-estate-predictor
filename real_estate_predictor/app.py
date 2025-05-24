from fastapi import FastAPI

from real_estate_predictor.predict import extract_input, predict

app = FastAPI()

# fastapi dev app.py
@app.get("/")
async def root():
    return {"msg": "please use the predict endpoint to generate predictions"}

@app.get("/predict/sale/{mlsNumber}")
async def predict_listing(mlsNumber, listing_type = "sale"):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}

@app.get("/predict/lease/{mlsNumber}")
async def predict_listing(mlsNumber, listing_type = "lease"):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}

@app.get("/health")
async def health():
    return {"status": "200 OK"}