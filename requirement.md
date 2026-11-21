# CardioPredict --- Project Requirements and Setup Guide

## 1. Project Overview

CardioPredict is a machine-learning-based heart disease prediction
project. It uses a trained Random Forest classifier and a preprocessing
pipeline, with a Streamlit web interface for entering patient
information and displaying a prediction.

## 2. Software Requirements

-   **Operating system:** macOS, Windows, or Linux
-   **Python:** Python 3.9 is recommended for this project because the
    model files were created in the project's Python 3.9 environment.
-   **Code editor:** Visual Studio Code or Jupyter Notebook
-   **Terminal:** macOS Terminal, Windows PowerShell, or a Linux
    terminal
-   **Web browser:** Google Chrome, Safari, Firefox, or another modern
    browser

Use the same Python and scikit-learn versions used when the model was
trained whenever possible. Scikit-learn model files saved with `joblib`
may not be compatible across different scikit-learn versions.

## 3. Python Libraries

The project uses these Python libraries:

  Library          Purpose
  ---------------- -------------------------------------------------------
  `pandas`         Loading and handling the heart disease dataset
  `numpy`          Numerical operations
  `scikit-learn`   Preprocessing, model training, and prediction
  `joblib`         Saving and loading the trained model and preprocessor
  `streamlit`      Running the prediction web application
  `matplotlib`     Creating visualizations
  `seaborn`        Statistical data visualizations

## 4. Required Project Files

Keep the following files in the same project folder:

``` text
heart-disease-prediction/
│
├── app.py
├── heart_disease_model.joblib
├── heart_disease_preprocessor.joblib
├── heart_disease_uci.csv
└── HDP.ipynb
```

-   `app.py` --- Streamlit application.
-   `heart_disease_model.joblib` --- trained Random Forest model.
-   `heart_disease_preprocessor.joblib` --- fitted preprocessing
    pipeline.
-   `heart_disease_uci.csv` --- dataset used for model development.
-   `HDP.ipynb` --- notebook containing data exploration, preprocessing,
    training, and evaluation.

The Streamlit app requires the two `.joblib` files. The CSV and notebook
are needed for reviewing or reproducing the model-development work, but
are not required just to launch the app.

## 5. Create and Activate a Virtual Environment

Open a terminal and navigate to the project folder.

### macOS / Linux

``` bash
cd path/to/heart-disease-prediction
python3.9 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

``` powershell
cd path\to\heart-disease-prediction
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
```

If Python 3.9 is not installed, install it first or use the Python
version used to train the saved model.

## 6. Install the Required Libraries

With the virtual environment activated, update pip:

``` bash
python -m pip install --upgrade pip
```

Install the libraries:

``` bash
python -m pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn
```

**Compatibility note:** For reliable loading, check the versions used to
train the model in the original notebook:

``` python
import sklearn
import joblib

print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
```

If the saved model was trained with specific versions, install those
same versions in the app environment. For example:

``` bash
python -m pip install scikit-learn==YOUR_TRAINING_VERSION joblib==YOUR_JOBLIB_VERSION
```

Replace the placeholders with the actual version numbers printed by the
training environment. Do not type the placeholder text literally.

## 7. Run the Web Application

Make sure the virtual environment is activated and the terminal is in
the project folder. Then run:

``` bash
python -m streamlit run app.py
```

Streamlit will print a local address, usually:

``` text
http://localhost:8501
```

Open that address in your web browser.

To stop the application, return to the terminal and press `Ctrl + C`.

## 8. Common Issues

### Model or preprocessor file not found

Check that both `.joblib` files are in the same folder as `app.py`, and
run Streamlit from that folder.

### Scikit-learn version or model-loading error

Use the same scikit-learn version that was used to train and save the
model. Activate the correct virtual environment before launching
Streamlit.

### `streamlit: command not found`

Install Streamlit in the active environment and launch it using:

``` bash
python -m pip install streamlit
python -m streamlit run app.py
```

### App opens with an error after clicking Predict

Check that the input DataFrame in `app.py` contains the same feature
names and compatible category values as the data used to fit the saved
preprocessor. The input feature names must match the training features.

## 9. Important Notice

This project is an academic demonstration of machine learning. Its
predictions are not a medical diagnosis and must not be used as a
substitute for assessment by a qualified healthcare professional.
