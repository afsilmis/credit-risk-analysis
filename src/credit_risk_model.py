# IMPORTS
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, brier_score_loss, classification_report, 
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
import joblib

# CONFIGURATION
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

GOOD_LOAN_STATUS = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
BAD_LOAN_STATUS = ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']

RISK_THRESHOLDS = [0, 0.11, 0.23, 0.43, 0.62, 1.0]
RISK_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

BEST_PARAMS = {
    'learning_rate': 0.010329,
    'max_depth': 9,
    'num_leaves': 54,
    'min_child_samples': 65,
    'min_split_gain': 0.328038,
    'subsample': 0.782749,
    'colsample_bytree': 0.801810,
    'reg_alpha': 0.003720,
    'reg_lambda': 0.000514,
    'n_estimators': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# DATA LOADING
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['target'] = df['loan_status'].apply(
        lambda x: 0 if x in GOOD_LOAN_STATUS else (1 if x in BAD_LOAN_STATUS else None)
    )
    df = df[df['target'].notna()].copy()
    print(f"✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

# FEATURE ENGINEERING
def safe_divide(numerator, denominator, fill_value=0):
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    return result

def create_features(df):
    df = df.copy()
    
    # Temporal features
    df['issue_d_dt'] = pd.to_datetime(df['issue_d'], format='%b-%y')
    df['issue_year'] = df['issue_d_dt'].dt.year
    df['issue_month'] = df['issue_d_dt'].dt.month
    df['issue_quarter'] = df['issue_d_dt'].dt.quarter
    
    # Credit history
    df['earliest_cr_line_dt'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
    df['credit_history_years'] = (pd.to_datetime('2016-01-01') - df['earliest_cr_line_dt']).dt.days / 365.25
    
    # Ratio features
    df['loan_to_income'] = safe_divide(df['loan_amnt'], df['annual_inc'])
    df['payment_to_income'] = safe_divide(df['installment'] * 12, df['annual_inc'])
    df['credit_utilization'] = safe_divide(df['revol_bal'], df['total_rev_hi_lim'])
    df['accounts_per_year'] = safe_divide(df['total_acc'], df['credit_history_years'])
    df['delinq_rate'] = safe_divide(df['delinq_2yrs'], df['total_acc'])
    
    # Categorical binning
    df['dti_range'] = pd.cut(df['dti'], bins=[0, 5, 10, 15, 20, 25, 50], labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25+'])
    df['util_range'] = pd.cut(df['revol_util'], bins=[0, 20, 40, 60, 80, 100, 200], labels=['0-20', '20-40', '40-60', '60-80', '80-100', '100+'])
    
    # Clean features
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').astype(float)
    df['home_ownership'] = df['home_ownership'].replace(['OTHER', 'ANY', 'NONE'], 'OTHER')
    
    return df

# PREPROCESSING
def build_preprocessor():
    num_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc', 'credit_history_years', 'loan_to_income', 'payment_to_income', 'credit_utilization']
    cat_cols = ['term', 'home_ownership', 'verification_status', 'issue_year', 'issue_month', 'grade', 'sub_grade']
    
    num_pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=RANDOM_STATE)),
        ('scaler', PowerTransformer(method='yeo-johnson'))
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')
    
    return preprocessor

# MODEL TRAINING
def train_model(X_train, y_train):
    model = LGBMClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)
    return model

def calibrate_model(base_model, X_train, y_train, method='isotonic'):
    calibrated = CalibratedClassifierCV(base_model, method=method, cv=5, n_jobs=-1)
    calibrated.fit(X_train, y_train)
    return calibrated

# EVALUATION
def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Brier Score': brier_score_loss(y_test, y_pred_proba)
    }
    
    print("\nMODEL PERFORMANCE")
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0,0]:5d} | FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d} | TP: {cm[1,1]:5d}")
    
    return metrics, y_pred_proba

# RISK SCORING
def assign_risk_tier(probabilities):
    tiers = pd.cut(probabilities, bins=RISK_THRESHOLDS, 
                   labels=RISK_LABELS, include_lowest=True)
    return tiers

def generate_tier_summary(y_true, y_pred_proba):
    tiers = assign_risk_tier(y_pred_proba)
    
    summary = pd.DataFrame({
        'Tier': RISK_LABELS,
        'Count': [sum(tiers == label) for label in RISK_LABELS],
        'Actual_Default_Rate': [y_true[tiers == label].mean() if sum(tiers == label) > 0 else 0 for label in RISK_LABELS],
        'Avg_Predicted_Prob': [y_pred_proba[tiers == label].mean() if sum(tiers == label) > 0 else 0 for label in RISK_LABELS]
    })
    
    summary['Percentage'] = (summary['Count'] / len(y_true) * 100).round(2)
    
    print("\nRISK TIER SUMMARY")
    print(summary.to_string(index=False))
    
    return summary

# VISUALIZATION
def plot_results(y_test, y_pred_proba, tier_summary):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc_score(y_test, y_pred_proba):.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].grid(alpha=0.3)
    
    # Tier Distribution
    axes[1, 0].bar(tier_summary['Tier'], tier_summary['Count'], 
                   color=['green', 'yellowgreen', 'gold', 'orange', 'red'], alpha=0.7)
    axes[1, 0].set_title('Tier Distribution')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Default Rate by Tier
    axes[1, 1].plot(tier_summary['Tier'], tier_summary['Actual_Default_Rate'] * 100, 
                    marker='o', linewidth=2)
    axes[1, 1].set_title('Actual Default Rate by Tier')
    axes[1, 1].set_ylabel('Default Rate (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

# MAIN PIPELINE
def main():
    print("\nCREDIT RISK SCORING MODEL - PIPELINE")
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    df = load_data('loan_data_2007_2014.csv')
    
    # 2. Feature engineering
    print("\n[2/7] Creating features...")
    df = create_features(df)
    print(f"✓ Features created: {df.shape[1]} columns")
    
    # 3. Drop leakage columns
    print("\n[3/7] Removing leakage columns...")
    leakage_cols = ['total_pymnt', 'total_rec_prncp', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'loan_status']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors='ignore')
    print(f"✓ Columns after cleanup: {df.shape[1]}")
    
    # 4. Split data
    print("\n[4/7] Splitting data...")
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # 5. Preprocessing & Training
    print("\n[5/7] Preprocessing & Training model...")
    preprocessor = build_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train, y_train)
    X_test_prep = preprocessor.transform(X_test)
    print(f"✓ Features after preprocessing: {X_train_prep.shape[1]}")
    
    base_model = train_model(X_train_prep, y_train)
    final_model = calibrate_model(base_model, X_train_prep, y_train)
    print("✓ Model trained and calibrated")
    
    # 6. Evaluation
    print("\n[6/7] Evaluating model...")
    metrics, y_pred_proba = evaluate_model(final_model, X_test_prep, y_test)
    
    # 7. Risk Scoring
    print("\n[7/7] Generating risk tiers...")
    tier_summary = generate_tier_summary(y_test, y_pred_proba)
    
    # Visualization
    print("\nGenerating visualizations...")
    plot_results(y_test, y_pred_proba, tier_summary)
    
    # Save model
    print("\nSaving model...")
    model_package = {
        'model': final_model,
        'preprocessor': preprocessor,
        'metrics': metrics,
        'tier_summary': tier_summary
    }
    joblib.dump(model_package, 'credit_risk_model.pkl')
    print("✓ Model saved to 'credit_risk_model.pkl'")
    
if __name__ == "__main__":
    main()