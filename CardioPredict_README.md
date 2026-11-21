# CardioPredict -- Heart Disease Prediction

CardioPredict is a machine learning project that predicts the likelihood
of heart disease from 13 patient health features. It uses a trained
Random Forest model and an interactive Streamlit application.

## Features

-   Accepts 13 patient health inputs.
-   Applies data preprocessing before prediction.
-   Compares four machine learning models.
-   Displays predictions through a Streamlit web app.

## Technologies

-   Python
-   Pandas and NumPy
-   Scikit-learn
-   Matplotlib and Seaborn
-   Streamlit
-   Joblib

## Models Evaluated

-   Logistic Regression
-   Support Vector Machine (SVM)
-   Decision Tree
-   Random Forest

**Selected model:** Random Forest\
**Test accuracy:** 86.41%

## Project Structure

``` text
heart-disease-prediction/
├── app.py
├── HDP.ipynb
├── heart_disease_uci.csv
├── heart_disease_model.joblib
├── heart_disease_preprocessor.joblib
├── requirements.txt
└── README.md
```

## Run Locally

Install the required packages:

``` bash
pip install -r requirements.txt
```

Start the application:

``` bash
streamlit run app.py
```

## Disclaimer

This project is developed for educational purposes only. It is not a
clinically validated medical diagnostic tool.
