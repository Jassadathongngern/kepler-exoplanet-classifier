# Kepler Exoplanet Classifier

## Project Overview

This project applies Machine Learning to astronomical data from the NASA Kepler Space Observatory. The objective is to develop an automated classification pipeline that distinguishes between Confirmed Exoplanets and False Positive signals using 11-dimensional transit telemetry data.

By analyzing light curves and transit parameters, this model serves as an automated screening tool to assist astronomers in validating exoplanet candidates efficiently.

## Key Features

* **Advanced ML Architecture** : Utilizes XGBoost (Extreme Gradient Boosting) for high-performance classification.
* **Hyperparameter Tuning** : Optimized via RandomizedSearchCV with 5-fold cross-validation to ensure model robustness and prevent overfitting.
* **Imbalanced Data Handling** : Implemented scale_pos_weight to manage the ratio between False Positives and Confirmed planets during training.
* **Interactive Dashboard** : A Streamlit-based web application for real-time astrophysical data validation and visualization.

## Tech Stack

* **Language** : Python 3.x
* **Machine Learning** : XGBoost, Scikit-learn
* **Data Manipulation** : Pandas, NumPy
* **Deployment** : Streamlit, Joblib

## Dataset Information

The model evaluates 11 critical astrophysical features from the Kepler Objects of Interest (KOI) dataset:

1. **Transit Signatures** : koi_period (Orbital period), koi_duration (Transit duration), koi_depth (Transit depth), koi_prad (Planetary radius).
2. **Stellar Properties** : koi_model_snr (Signal-to-Noise), koi_teq (Equilibrium Temp), koi_srad (Stellar Radius), koi_steff (Stellar Temp).
3. **Advanced Physics** : koi_slogg (Surface Gravity), koi_insol (Insolation Flux), koi_kepmag (Kepler Magnitude).

## Model Evaluation

The model achieved high performance in distinguishing confirmed exoplanets from astrophysical false positives:

* **Overall Accuracy** : 92%
* **False Positive Precision** : 0.95 (High reliability in filtering out noise)
* **Confirmed Planet Recall** : 0.91 (Strong capability in detecting actual exoplanets)

## Installation & Execution

1. Clone the repository:
   **Bash**

   ```
   git clone https://github.com/Jassadathongngern/kepler-exoplanet-classifier.git
   cd kepler-exoplanet-classifier
   ```
2. Install dependencies:
   **Bash**

   ```
   pip install pandas scikit-learn xgboost streamlit joblib
   ```
3. Train the model:
   **Bash**

   ```
   python src/train_model.py
   ```
4. Launch the dashboard:
   **Bash**

   ```
   streamlit run app.py
   ```
