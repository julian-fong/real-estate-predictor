"""Test file to generate predictions."""

from real_estate_predictor.predict import extract_input, predict, predict_sale_listing

mlsNumber = "N12174739"


def test_sale_listing_prediction():
    """Test sale listing prediction."""

    prediction = predict_sale_listing(mlsNumber)
    assert isinstance(prediction, dict)
    assert "mlsNumber" in prediction
    assert "prediction" in prediction
    assert prediction["mlsNumber"] == mlsNumber


def test_predict_with_sale():
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, "sale")[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}
