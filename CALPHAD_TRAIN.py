#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("CALPHAD_featurized_compositions.csv")
labels = pd.read_excel("CalPhad_700C.xlsx")[['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']].fillna(0)

# Features
feature_columns = df.columns[-145:]
X = df[feature_columns]


y_df = labels
phase_targets = ['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']

# Model factory
def create_pipeline(model_type):
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=500, max_features=0.33, n_jobs=-1, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                             tree_method='hist', n_jobs=-1, random_state=42)
    elif model_type == 'gpr':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Scorers
scoring = {
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score)
}

# 1. Cross-validation
print("=== CROSS-VALIDATION RESULTS ===")
all_metrics = []

for phase in phase_targets:
    y = y_df[phase]
    for model_type in ['rf', 'xgb', 'gpr']:
        pipe = create_pipeline(model_type)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_validate(pipe, X, y, cv=kfold, scoring=scoring, n_jobs=-1)

        metrics = {
            'Model': model_type.upper(),
            'Phase': phase,
            'CV_MAE_mean': np.mean(cv_results['test_MAE']),
            'CV_MAE_std': np.std(cv_results['test_MAE']),
            'CV_R2_mean': np.mean(cv_results['test_R2']),
            'CV_R2_std': np.std(cv_results['test_R2'])
        }
        all_metrics.append(metrics)

        print(f"\n{model_type.upper()} ({phase}) - 10-Fold CV Results:")
        print(f"MAE: {metrics['CV_MAE_mean']:.4f} ± {metrics['CV_MAE_std']:.4f}")
        print(f"R² : {metrics['CV_R2_mean']:.4f} ± {metrics['CV_R2_std']:.4f}")

# 2. Final training on full data
print("\n=== FINAL TRAINING ON FULL DATA ===")
final_results = []
plot_data = []

for phase in phase_targets:
    y = y_df[phase]
    for model_type in ['rf', 'xgb', 'gpr']:
        model_name = f"{model_type}_{phase.replace('@','').replace('(','').replace(')','')}"
        print(f"\nTraining {model_name}...")

        pipe = create_pipeline(model_type)
        pipe.fit(X, y)
        joblib.dump(pipe, f"{model_name}_model.pkl")
        print(f"Saved model as {model_name}_model.pkl")

        y_pred = pipe.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        final_results.append({
            'Model': model_type.upper(),
            'Phase': phase,
            'MAE': mae,
            'R2': r2,
            'Samples': len(X)
        })

        plot_data.append(pd.DataFrame({
            'True': y,
            'Predicted': y_pred,
            'Model': model_type.upper(),
            'Phase': phase,
            'R2': r2
        }))

        print(f"MAE: {mae:.6f}, R²: {r2:.6f}")

# 3. Save summary
cv_df = pd.DataFrame(all_metrics)
full_df = pd.DataFrame(final_results)
cv_df.to_csv("calphad_cv_results.csv", index=False)
full_df.to_csv("calphad_full_results.csv", index=False)
print("\nSaved performance summaries to 'calphad_cv_results.csv' and 'calphad_full_results.csv'")


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("CALPHAD_featurized_compositions.csv")
labels = pd.read_excel("CalPhad_700C.xlsx")[['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']].fillna(0)

# Features
#feature_columns = df.columns[-145:]
#X = df[feature_columns]

reduced_features = pd.read_csv("reduced_feature_list4.csv", header=None).squeeze("columns").dropna().astype(str).tolist()
X = df[reduced_features]

y_df = labels
phase_targets = ['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']

# Model factory
def create_pipeline(model_type):
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=500, max_features=0.33, n_jobs=-1, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                             tree_method='hist', n_jobs=-1, random_state=42)
    elif model_type == 'gpr':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Scorers
scoring = {
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score)
}

# 1. Cross-validation
print("=== CROSS-VALIDATION RESULTS ===")
all_metrics = []

for phase in phase_targets:
    y = y_df[phase]
    print(y)
    for model_type in ['rf', 'xgb', 'gpr']:
        pipe = create_pipeline(model_type)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_validate(pipe, X, y, cv=kfold, scoring=scoring, n_jobs=-1)

        metrics = {
            'Model': model_type.upper(),
            'Phase': phase,
            'CV_MAE_mean': np.mean(cv_results['test_MAE']),
            'CV_MAE_std': np.std(cv_results['test_MAE']),
            'CV_R2_mean': np.mean(cv_results['test_R2']),
            'CV_R2_std': np.std(cv_results['test_R2'])
        }
        all_metrics.append(metrics)

        print(f"\n{model_type.upper()} ({phase}) - 10-Fold CV Results:")
        print(f"MAE: {metrics['CV_MAE_mean']:.4f} ± {metrics['CV_MAE_std']:.4f}")
        print(f"R² : {metrics['CV_R2_mean']:.4f} ± {metrics['CV_R2_std']:.4f}")

# 2. Final training on full data
print("\n=== FINAL TRAINING ON FULL DATA ===")
final_results = []
plot_data = []

for phase in phase_targets:
    y = y_df[phase]
    for model_type in ['rf', 'xgb', 'gpr']:
        model_name = f"{model_type}_{phase.replace('@','').replace('(','').replace(')','')}"
        print(f"\nTraining {model_name}...")

        pipe = create_pipeline(model_type)
        pipe.fit(X, y)
        joblib.dump(pipe, f"{model_name}_model.pkl")
        print(f"Saved model as {model_name}_model.pkl")

        y_pred = pipe.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        final_results.append({
            'Model': model_type.upper(),
            'Phase': phase,
            'MAE': mae,
            'R2': r2,
            'Samples': len(X)
        })

        plot_data.append(pd.DataFrame({
            'True': y,
            'Predicted': y_pred,
            'Model': model_type.upper(),
            'Phase': phase,
            'R2': r2
        }))

        print(f"MAE: {mae:.6f}, R²: {r2:.6f}")

# 3. Save summary
cv_df = pd.DataFrame(all_metrics)
full_df = pd.DataFrame(final_results)
cv_df.to_csv("calphad_cv_results_reduced.csv", index=False)
full_df.to_csv("calphad_full_results_reduced.csv", index=False)
print("\nSaved performance summaries to 'calphad_cv_results_reduced.csv' and 'calphad_full_results_reduced.csv'")


