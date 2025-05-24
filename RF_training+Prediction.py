#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score
import joblib

# Load data
df_conf = pd.read_csv("CALPHAD_featurized_compositions.csv")
#labels_df = pd.read_excel("try.xlsx")  # Contains f(@Bcc), f(@Fcc), f(@Sigma)
labels_df = pd.read_excel("CalPhad_700C.xlsx")

# Define features and labels
feature_columns = df_conf.columns[-145:]
X = df_conf[feature_columns]
y = labels_df[['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']].fillna(0)

# Scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))

# 10-fold cross-validation for RMSE
cv = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-rmse_scores)

# 10-fold cross-validation for R² (for each output separately)
r2_scores = []
for i, target in enumerate(['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']):
    r2 = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                         X_scaled, y.iloc[:, i],
                         cv=cv, scoring='r2')
    r2_scores.append(r2)

# Print results
print("✅ Random Forest Multi-Output Regression")
print(f"10-fold RMSE: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}\n")

phase_names = ['f(@Bcc)', 'f(@Fcc)', 'f(@Sigma)']
for i, scores in enumerate(r2_scores):
    print(f"📊 R² for {phase_names[i]}: {scores.mean():.4f} ± {scores.std():.4f}")

# Train on full data and save
rf.fit(X_scaled, y)
joblib.dump(rf, "rf_phase_fraction_model.pkl")
joblib.dump(scaler, "rf_feature_scaler.pkl")

print("\n📦 Model saved as 'rf_phase_fraction_model.pkl'")
print("📦 Scaler saved as 'rf_feature_scaler.pkl'")


# In[ ]:


import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("rf_phase_fraction_model.pkl")
scaler = joblib.load("rf_feature_scaler.pkl")

# Define path to datasets
files = [
    "ternary_compositions_featurized101.csv"
]

# Define feature columns (assuming the last 145 columns match training)
for file in files:
    print(f"\n🔍 Processing {file}")
    
    # Load data
    df = pd.read_csv(file)
    
    # Select feature columns (last 145)
    feature_columns = df.columns[-145:]
    X = df[feature_columns]
    
    # Apply the same scaling
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    df[['pred_f(@Bcc)', 'pred_f(@Fcc)', 'pred_f(@Sigma)']] = predictions
    
    # Save result
    output_file = file.replace(".csv", "_with_predictions.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import ternary
from matplotlib.patches import Patch
import numpy as np

# File paths
files = [
    "ternary_compositions_featurized101_with_predictions.csv"
]

# Phase color mapping
phase_colors = {
    'pred_f(@Fcc)': 'blue',
    'pred_f(@Bcc)': 'red',
    'pred_f(@Sigma)': 'orange'
}

for file in files:
    df = pd.read_csv(file)

    # Compute ternary coordinates: A = Fe+Ni, B = Mn, C = Cr
    A = df['Fe'] + df['Ni']
    B = df['Mn']
    C = df['Cr']
    total = A + B + C

    # Rule out edge points (A == 0 or B == 0 or C == 0)
    mask = (A > 0) & (B > 0) & (C > 0)
    df = df[mask].reset_index(drop=True)
    A = A[mask].reset_index(drop=True)
    B = B[mask].reset_index(drop=True)
    C = C[mask].reset_index(drop=True)
    total = total[mask].reset_index(drop=True)

    coords = np.stack([B / total * 100, C / total * 100, A / total * 100], axis=1)

    # Setup ternary plot with square aspect ratio
    fig, tax = ternary.figure(scale=100)
    #fig.set_size_inches(7,6.06)  # ensures equilateral triangle: height = sqrt(3)/2 * width
    tax.boundary(linewidth=2)
    tax.gridlines(color="black", multiple=10, linewidth=0.5)
    tax.left_axis_label("Fe+Ni", fontsize=12, offset=0.16)
    tax.right_axis_label("Cr", fontsize=12, offset=0.16)
    tax.bottom_axis_label("Mn", fontsize=12, offset=0.06)
    tax.ticks(axis='lbr', multiple=10, linewidth=1, tick_formats="%.0f", offset=0.02)
    #tax.clear_matplotlib_ticks()

    # Plot predictions with transparency
    for phase, color in phase_colors.items():
        if phase in df.columns:
            alphas = df[phase].clip(0, 1).to_numpy()
            for i, alpha in enumerate(alphas):
                if alpha > 0:
                    alpha = min(alpha * 1, 1.0)  # double opacity
                    tax.scatter([tuple(coords[i])], color=color, alpha=alpha, s=60)

    # Legend
    legend_handles = [Patch(color=c, label=ph) for ph, c in phase_colors.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    tax.clear_matplotlib_ticks()
    #tax.ticks(axis='lbr', linewidth=1, multiple=10)
    tax.get_axes().axis('off')

    # Save and close
    output_file = file.replace(".csv", "_ternary_plot_custom_axes.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Saved ternary diagram to: {output_file}")

