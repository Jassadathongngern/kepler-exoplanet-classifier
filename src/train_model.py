import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import os

def main():
    print("[INFO] Loading Kepler cumulative dataset...")
    features = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_model_snr', 'koi_teq', 'koi_srad', 
        'koi_steff', 'koi_slogg', 'koi_insol',
        'koi_kepmag'
    ]
    target = 'koi_disposition'

    try:
        df = pd.read_csv('data/cumulative.csv', usecols=features + [target])
    except FileNotFoundError:
        print("[ERROR] Dataset not found.")
        return

    df = df.dropna()
    df = df[df[target] != 'CANDIDATE']
    df['is_planet'] = df[target].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

    X = df[features]
    y = df['is_planet']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Initiating Maximum Capacity XGBoost Tuning... (This is the endgame!)")
    param_dist = {
        'n_estimators': [500, 1000, 1500],        
        'learning_rate': [0.01, 0.05, 0.1],        
        'max_depth': [6, 8, 10, 12],               
        'subsample': [0.6, 0.8, 1.0],              
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1.0],              
        'min_child_weight': [1, 3, 5]              
    }

    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
    base_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', scale_pos_weight=ratio)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,          
        scoring='f1',       
        cv=5,               
        verbose=2,          
        random_state=42,
        n_jobs=-1           
    )

    search.fit(X_train, y_train)

    print("\n[SUCCESS] Best Ultra-XGBoost Parameters:")
    print(search.best_params_)

    best_model = search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    print("\n--- Absolute Maximum Evaluation Report ---")
    print(classification_report(y_test, y_pred, target_names=['False Positive (0)', 'Confirmed Planet (1)']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/kepler_model.pkl')
    print("\n[SUCCESS] The Absolute Best XGBoost model saved successfully.")

if __name__ == '__main__':
    main()