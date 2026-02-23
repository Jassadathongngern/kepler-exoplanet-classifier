# Kepler Exoplanet Classifier 🪐

## Project Overview

This project applies Machine Learning to astronomical data collected by the NASA Kepler Space Observatory. The objective is to develop an automated classification pipeline that distinguishes between confirmed exoplanets and false positive signals (such as eclipsing binaries or instrumental noise) using transit photometry data.

By analyzing light curves and transit parameters, this model serves as an automated screening tool to assist astronomers in validating exoplanet candidates efficiently.

## Key Features

* **Real-World Data Processing:** Utilizes the official Kepler cumulative dataset (`cumulative.csv`) containing over 10,000 objects of interest.
* **Automated Data Cleaning:** Implements strict data preprocessing by isolating ground-truth labels (`CONFIRMED` and `FALSE POSITIVE`) and discarding ambiguous candidates.
* **Predictive Modeling:** Deploys a Random Forest Classifier to achieve high-accuracy binary classification.
* **Interactive Dashboard:** Features a Streamlit-based web application allowing users to input transit parameters and receive real-time classification probabilities.

## Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForestClassifier, train_test_split, classification_report)
* **Web Framework:** Streamlit
* **Model Deployment:** Joblib

## Dataset Information

The dataset consists of Kepler Objects of Interest (KOI). The model evaluates the following critical astrophysical features to make predictions:

1. `koi_period`: Orbital period (days) - The time it takes for the object to complete one orbit.
2. `koi_duration`: Transit duration (hours) - The length of time the object blocks the host star's light.
3. `koi_depth`: Transit depth (parts per million) - The fraction of stellar flux lost at the minimum of the planetary transit.
4. `koi_prad`: Planetary radius (Earth radii) - The calculated physical size of the object.

## Machine Learning Pipeline

1. **Data Ingestion & Filtering:** Loaded the raw dataset and filtered out unverified `CANDIDATE` records.
2. **Feature Engineering:** Mapped target variables into binary classes (1 for Confirmed, 0 for False Positive).
3. **Model Training:** Utilized an ensemble learning method (Random Forest) with 200 estimators to prevent overfitting and capture complex, non-linear astronomical relationships.
4. **Evaluation:** Assessed the model using Precision, Recall, and F1-Score to ensure a balanced detection capability.

## Installation & Execution

Follow these steps to run the project locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/kepler-exoplanet-classifier.git](https://github.com/your-username/kepler-exoplanet-classifier.git)
   cd kepler-exoplanet-classifier
   ```


* Install the required dependencies:
  **Bash**

  ```
  pip install pandas scikit-learn streamlit joblib
  ```
* Train the model (this will generate the `kepler_model.pkl` file):
  **Bash**

  ```
  python src/train_kepler.py
  ```
* Launch the Streamlit dashboard:
  **Bash**

  ```
  streamlit run app.py
  ```
