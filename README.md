# Diabetes Risk Assessor Web App

A Streamlit web application based on `diabetesPredictor.ipynb`.

## Run locally

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app automatically loads `diabetes.csv` from the same repository. Users do not upload the dataset.

## Streamlit Community Cloud

Push these files to the root of a GitHub repository:

- `app.py`
- `diabetes.csv`
- `requirements.txt`
- `README.md`

In Streamlit Community Cloud, select the repository, branch `main`, and `app.py` as the main file.

## Model logic

The supplied model pipeline is retained: `glyhb >= 6.5` creates the target, excluded columns are removed, numerical values use median imputation, categorical values use mode imputation, categorical variables are one-hot encoded, the data is split with `test_size=0.2`, `random_state=42`, and stratification, SMOTE is applied to the training data, features are standardized, Logistic Regression uses `max_iter=1000`, and the final prediction uses a 0.45 probability threshold.

The public interface does not expose model evaluation metrics or model-download controls.

This application is a demonstration of the supplied machine-learning notebook and is not a medical diagnostic system.
