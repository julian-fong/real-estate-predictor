<h1 align="center">
<span><i>Real Estate Predictor</i></span>
</h1>

The Real Estate Predictor is a library designed to estimate property prices using various features such as location, size, and amenities in the Greater Toronto Area. By leveraging machine learning algorithms, this tool aims to provide accurate price predictions to assist buyers, sellers, and real estate professionals. With this library, you have the capabilities to do data cleaning, data pre-processing, and feature engineering and machine model training seamlessly through the custom built features. See the example directory for more.

## ⚠️ Important

This project is also built using the **Repliers** API. To utilize the dataset modules to extract live GTA listings data, or to do predictions via the sample models in this library, you will need a key to do so. More information about Repliers can be found [here](https://repliers.com/).

## Installation

`real-estate-predictor` supports Python 3.10 and greater, and you can install the library by downloading it from the github repository via the pip command:

```
git clone https://github.com/julian-fong/real-estate-predictor.git
cd real-estate-predictor
pip install .
```

or via 

```
pip install git+https://github.com/julian-fong/real-estate-predictor.git
```

## Features

Data Cleaning: Handles missing values, removing bad/invalid values via `pandas`

Feature Engineering: Creates new features using existing predictors via `pandas`

Data Preprocessing: Scaling numerical predictors, encodes categorical variables, and splits the dataset into training, validation, and test sets via `sklearn`.

Model Training: Implements regression models like Linear Regression, and XGBoost.

Model Evaluation: Provides metrics such as RMSE, MAE, and R² to assess model performance.

Docker Support: Includes a Dockerfile for containerized deployment.

Continuous Integration: Configured with GitHub Actions for automated testing and deployment.