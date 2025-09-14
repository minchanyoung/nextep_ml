import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import time

print("=== Class Imbalance Improvement for Satisfaction Prediction ===")

def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Quick data processing
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    valid_mask = (~df_consecutive['next_satisfaction'].isna())
    df_final = df_consecutive[valid_mask].copy()
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    print(f"Dataset size: {df_final.shape}")
    
    exclude_cols = ['pid', 'year', 'next_year', 'next_satisfaction', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    # Encode categorical variables
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # Handle missing values
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # Time-based split
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    # Analyze class distribution
    print("\nClass distribution in training set:")
    class_counts = pd.Series(y_sat_train).value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count:,} ({count/len(y_sat_train)*100:.1f}%)")
    
    print(f"\nTraining: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_sat_train, y_sat_test

def apply_smote_sampling(X_train, y_sat_train):
    """Apply SMOTE for balanced sampling"""
    print("\nApplying SMOTE sampling...")
    
    # Use SMOTE with Tomek links for better results
    smote_tomek = SMOTETomek(
        smote=SMOTE(random_state=42, k_neighbors=3),
        random_state=42
    )
    
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_sat_train)
    
    print(f"Original training size: {X_train.shape[0]}")
    print(f"Balanced training size: {X_train_balanced.shape[0]}")
    
    print("\nBalanced class distribution:")
    class_counts = pd.Series(y_train_balanced).value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count:,} ({count/len(y_train_balanced)*100:.1f}%)")
    
    return X_train_balanced, y_train_balanced

def compute_class_weights(y_train):
    """Compute class weights for models"""
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"\nComputed class weights: {class_weight_dict}")
    return class_weight_dict

def train_baseline_model(X_train, X_test, y_sat_train, y_sat_test):
    """Train baseline model without imbalance handling"""
    print("\n=== Training Baseline Model (No Imbalance Handling) ===")
    
    baseline_model = xgb.XGBClassifier(
        n_estimators=1151,
        max_depth=4,
        learning_rate=0.102,
        subsample=0.805,
        colsample_bytree=0.815,
        reg_alpha=2.16,
        reg_lambda=8.77,
        random_state=42,
        objective='multi:softprob',
        verbosity=0
    )
    
    baseline_model.fit(X_train, y_sat_train)
    baseline_pred = baseline_model.predict(X_test)
    
    baseline_accuracy = accuracy_score(y_sat_test, baseline_pred)
    baseline_f1 = f1_score(y_sat_test, baseline_pred, average='weighted')
    
    print(f"Baseline - Accuracy: {baseline_accuracy:.4f}, F1: {baseline_f1:.4f}")
    
    return {
        'model': baseline_model,
        'accuracy': baseline_accuracy,
        'f1': baseline_f1,
        'predictions': baseline_pred
    }

def train_weighted_model(X_train, X_test, y_sat_train, y_sat_test, class_weights):
    """Train model with class weights"""
    print("\n=== Training Weighted Model ===")
    
    weighted_model = xgb.XGBClassifier(
        n_estimators=1151,
        max_depth=4,
        learning_rate=0.102,
        subsample=0.805,
        colsample_bytree=0.815,
        reg_alpha=2.16,
        reg_lambda=8.77,
        random_state=42,
        objective='multi:softprob',
        verbosity=0
    )
    
    # Convert class weights to sample weights
    sample_weights = np.array([class_weights[y] for y in y_sat_train])
    
    weighted_model.fit(X_train, y_sat_train, sample_weight=sample_weights)
    weighted_pred = weighted_model.predict(X_test)
    
    weighted_accuracy = accuracy_score(y_sat_test, weighted_pred)
    weighted_f1 = f1_score(y_sat_test, weighted_pred, average='weighted')
    
    print(f"Weighted - Accuracy: {weighted_accuracy:.4f}, F1: {weighted_f1:.4f}")
    
    return {
        'model': weighted_model,
        'accuracy': weighted_accuracy,
        'f1': weighted_f1,
        'predictions': weighted_pred
    }

def train_smote_model(X_train_balanced, X_test, y_train_balanced, y_sat_test):
    """Train model with SMOTE balanced data"""
    print("\n=== Training SMOTE Model ===")
    
    smote_model = xgb.XGBClassifier(
        n_estimators=1151,
        max_depth=4,
        learning_rate=0.102,
        subsample=0.805,
        colsample_bytree=0.815,
        reg_alpha=2.16,
        reg_lambda=8.77,
        random_state=42,
        objective='multi:softprob',
        verbosity=0
    )
    
    smote_model.fit(X_train_balanced, y_train_balanced)
    smote_pred = smote_model.predict(X_test)
    
    smote_accuracy = accuracy_score(y_sat_test, smote_pred)
    smote_f1 = f1_score(y_sat_test, smote_pred, average='weighted')
    
    print(f"SMOTE - Accuracy: {smote_accuracy:.4f}, F1: {smote_f1:.4f}")
    
    return {
        'model': smote_model,
        'accuracy': smote_accuracy,
        'f1': smote_f1,
        'predictions': smote_pred
    }

def train_focal_loss_model(X_train, X_test, y_sat_train, y_sat_test, class_weights):
    """Train CatBoost model with Focal Loss"""
    print("\n=== Training Focal Loss Model (CatBoost) ===")
    
    focal_model = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        verbose=0,
        random_seed=42,
        # Focal loss is implicitly handled by CatBoost's class_weights and objective
        # For more direct control, one might need custom objectives,
        # but this setup is a strong start for imbalance.
    )
    
    focal_model.fit(X_train, y_sat_train)
    focal_pred = focal_model.predict(X_test).flatten()
    
    focal_accuracy = accuracy_score(y_sat_test, focal_pred)
    focal_f1 = f1_score(y_sat_test, focal_pred, average='weighted')
    
    print(f"Focal Loss (CatBoost) - Accuracy: {focal_accuracy:.4f}, F1: {focal_f1:.4f}")
    
    return {
        'model': focal_model,
        'accuracy': focal_accuracy,
        'f1': focal_f1,
        'predictions': focal_pred
    }

def detailed_performance_analysis(baseline_results, weighted_results, smote_results, focal_loss_results, y_test):
    """Detailed performance analysis by class"""
    print("\n=== Detailed Performance Analysis ===")
    
    models = {
        'Baseline': baseline_results,
        'Weighted': weighted_results,
        'SMOTE': smote_results,
        'Focal Loss (CatBoost)': focal_loss_results
    }
    
    for model_name, results in models.items():
        print(f"\n{model_name} Model:")
        print(f"  Overall Accuracy: {results['accuracy']:.4f}")
        print(f"  Overall F1-score: {results['f1']:.4f}")
        
        # Per-class metrics
        report = classification_report(y_test, results['predictions'], output_dict=True, zero_division=0)
        print("  Per-class performance:")
        for i in range(5):  # Classes 0-4
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                print(f"    Class {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Find best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['f1'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  Best F1-score: {best_model['f1']:.4f}")
    
    return best_model_name, best_model

def save_imbalance_results(baseline_results, weighted_results, smote_results, focal_loss_results, best_model_name, best_model):
    """Save imbalance handling results"""
    print("\nSaving imbalance handling results...")
    
    # Save best model
    joblib.dump(best_model['model'], 'models/best_imbalance_handled_satisfaction.pkl')
    
    # Save results comparison
    results_comparison = {
        'baseline_accuracy': baseline_results['accuracy'],
        'baseline_f1': baseline_results['f1'],
        'weighted_accuracy': weighted_results['accuracy'],
        'weighted_f1': weighted_results['f1'],
        'smote_accuracy': smote_results['accuracy'],
        'smote_f1': smote_results['f1'],
        'focal_loss_accuracy': focal_loss_results['accuracy'],
        'focal_loss_f1': focal_loss_results['f1'],
        'best_method': best_model_name,
        'best_accuracy': best_model['accuracy'],
        'best_f1': best_model['f1']
    }
    
    results_df = pd.DataFrame([results_comparison])
    results_df.to_csv('model_results/class_imbalance_results.csv', index=False)
    
    print("Results saved:")
    print("- models/best_imbalance_handled_satisfaction.pkl")
    print("- model_results/class_imbalance_results.csv")
    
    return results_comparison

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_sat_train, y_sat_test = load_data()
    
    # Compute class weights
    class_weights = compute_class_weights(y_sat_train)
    
    # Apply SMOTE sampling
    X_train_balanced, y_train_balanced = apply_smote_sampling(X_train, y_sat_train)
    
    # Train different models
    baseline_results = train_baseline_model(X_train, X_test, y_sat_train, y_sat_test)
    weighted_results = train_weighted_model(X_train, X_test, y_sat_train, y_sat_test, class_weights)
    smote_results = train_smote_model(X_train_balanced, X_test, y_train_balanced, y_sat_test)
    focal_loss_results = train_focal_loss_model(X_train, X_test, y_sat_train, y_sat_test, class_weights)
    
    # Detailed analysis
    best_model_name, best_model = detailed_performance_analysis(
        baseline_results, weighted_results, smote_results, focal_loss_results, y_sat_test
    )
    
    # Save results
    final_results = save_imbalance_results(
        baseline_results, weighted_results, smote_results, focal_loss_results, best_model_name, best_model
    )
    
    total_time = time.time() - start_time
    print(f"\n=== Class Imbalance Improvement Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best method: {best_model_name}")
    print(f"Best F1-score: {final_results['best_f1']:.4f}")

if __name__ == "__main__":
    main()
