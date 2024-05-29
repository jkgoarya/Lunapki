import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import logging
import os
from imblearn.over_sampling import SMOTE
from ta import add_all_ta_features
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import optuna
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(filename='train_model.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

def load_preprocessed_data(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Loaded preprocessed data from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error reading preprocessed data from {file_path}: {e}")
            return pd.DataFrame()
    else:
        logging.error(f"File {file_path} does not exist or is empty")
        return pd.DataFrame()

# Load preprocessed data
data = load_preprocessed_data('preprocessed_data.csv')
if data.empty:
    raise ValueError("Preprocessed data is empty. Please check the preprocessing step.")

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Add additional technical indicators using the ta library
data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)

# Define features and target
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # Target: 1 if next day's close is higher, else 0
features = data.columns.drop(['Target'])  # Use all columns except the target as features
X = data[features]
y = data['Target']

# Feature Selection
selector = SelectKBest(f_classif, k=20)
X = selector.fit_transform(X, y)
selected_feature_names = data[features].columns[selector.get_support()]

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define algorithms and hyperparameter grids
algorithms = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 1.0, 1.5]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.01, 0.1, 1.0, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.3, 0.7]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'boosting_type': ['gbdt', 'dart'],
            'min_data_in_leaf': [20, 50, 100]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, silent=True),
        'params': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [3, 5, 7, 9],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
    }
}

best_models = {}

# Perform randomized search and evaluate models
for name, algo in algorithms.items():
    logging.info(f"Training {name} model...")
    rand_search = RandomizedSearchCV(estimator=algo['model'], param_distributions=algo['params'], n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
    rand_search.fit(X_train_scaled, y_train)
    best_model = rand_search.best_estimator_
    best_models[name] = best_model
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    logging.info(f"{name} Cross-Validation Scores: {cv_scores}")
    
    # Evaluate the model
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    logging.info(f"{name} Model Accuracy: {accuracy}")
    logging.info(f"{name} Model Precision: {precision}")
    logging.info(f"{name} Model Recall: {recall}")
    logging.info(f"{name} Model F1 Score: {f1}")
    logging.info(f"{name} Model ROC AUC: {roc_auc}")

    logging.info(f"{name} Model Classification Report: \n{classification_report(y_test, y_pred)}")
    logging.info(f"{name} Model Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

    print(f"{name} Model Accuracy: {accuracy}")
    print(f"{name} Model Precision: {precision}")
    print(f"{name} Model Recall: {recall}")
    print(f"{name} Model F1 Score: {f1}")
    print(f"{name} Model ROC AUC: {roc_auc}")
    print(f"{name} Model Classification Report: \n{classification_report(y_test, y_pred)}")
    print(f"{name} Model Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

    # Save the model
    joblib.dump(best_model, f'{name}_model.pkl')

# Ensemble Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_models['RandomForest']),
    ('gb', best_models['GradientBoosting']),
    ('ab', best_models['AdaBoost']),
    ('lr', best_models['LogisticRegression']),
    ('xgb', best_models['XGBoost']),
    ('lgbm', best_models['LightGBM']),
    ('catboost', best_models['CatBoost'])
], voting='soft')

voting_clf.fit(X_train_scaled, y_train)

# Evaluate the ensemble model
y_pred_ensemble = voting_clf.predict(X_test_scaled)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble)

logging.info(f"Ensemble Model Accuracy: {accuracy_ensemble}")
logging.info(f"Ensemble Model Precision: {precision_ensemble}")
logging.info(f"Ensemble Model Recall: {recall_ensemble}")
logging.info(f"Ensemble Model F1 Score: {f1_ensemble}")
logging.info(f"Ensemble Model ROC AUC: {roc_auc_ensemble}")

print(f"Ensemble Model Accuracy: {accuracy_ensemble}")
print(f"Ensemble Model Precision: {precision_ensemble}")
print(f"Ensemble Model Recall: {recall_ensemble}")
print(f"Ensemble Model F1 Score: {f1_ensemble}")
print(f"Ensemble Model ROC AUC: {roc_auc_ensemble}")

# Save the ensemble model
joblib.dump(voting_clf, 'ensemble_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Model Interpretability with SHAP
explainer = shap.TreeExplainer(voting_clf)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled, feature_names=selected_feature_names)

plt.title(f"SHAP Summary Plot for Ensemble Model")
plt.show()
