from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from real_estate_predictor.predict import extract_input, predict

app = FastAPI()

# fastapi dev app.py

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://landpower.ca",
    "https://www.landpower.ca",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"msg": "please use the predict endpoint to generate predictions"}


@app.get("/predict/sale/{mlsNumber}")
async def predict_sale_listing(mlsNumber, listing_type="sale"):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}


@app.get("/predict/lease/{mlsNumber}")
async def predict_lease_listing(mlsNumber, listing_type="lease"):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}


@app.get("/health")
async def health():
    return {"status": "200 OK"}
