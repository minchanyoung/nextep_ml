import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import time

print("=== Simple Class Weight Improvement ===")

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
    total_samples = len(y_sat_train)
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count:,} ({count/total_samples*100:.1f}%)")
    
    print(f"\nTraining: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_sat_train, y_sat_test

def train_baseline_model(X_train, X_test, y_sat_train, y_sat_test):
    """Train baseline model without class weighting"""
    print("\n=== Training Baseline Model ===")
    
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
    baseline_f1_macro = f1_score(y_sat_test, baseline_pred, average='macro')
    
    print(f"Baseline - Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline - F1 (weighted): {baseline_f1:.4f}")
    print(f"Baseline - F1 (macro): {baseline_f1_macro:.4f}")
    
    return {
        'model': baseline_model,
        'accuracy': baseline_accuracy,
        'f1_weighted': baseline_f1,
        'f1_macro': baseline_f1_macro,
        'predictions': baseline_pred
    }

def train_weighted_models(X_train, X_test, y_sat_train, y_sat_test):
    """Train models with different class weighting strategies"""
    print("\n=== Training Weighted Models ===")
    
    # Compute class weights
    classes = np.unique(y_sat_train)
    class_weights_balanced = compute_class_weight('balanced', classes=classes, y=y_sat_train)
    class_weight_dict = dict(zip(classes, class_weights_balanced))
    
    print(f"Computed balanced class weights: {class_weight_dict}")
    
    # Model 1: Balanced class weights
    print("\nTraining with balanced class weights...")
    sample_weights = np.array([class_weight_dict[y] for y in y_sat_train])
    
    weighted_model1 = xgb.XGBClassifier(
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
    
    weighted_model1.fit(X_train, y_sat_train, sample_weight=sample_weights)
    weighted_pred1 = weighted_model1.predict(X_test)
    
    weighted_accuracy1 = accuracy_score(y_sat_test, weighted_pred1)
    weighted_f1_1 = f1_score(y_sat_test, weighted_pred1, average='weighted')
    weighted_f1_macro1 = f1_score(y_sat_test, weighted_pred1, average='macro')
    
    print(f"Weighted (balanced) - Accuracy: {weighted_accuracy1:.4f}")
    print(f"Weighted (balanced) - F1 (weighted): {weighted_f1_1:.4f}")
    print(f"Weighted (balanced) - F1 (macro): {weighted_f1_macro1:.4f}")
    
    # Model 2: Custom class weights (more aggressive for minority classes)
    print("\nTraining with custom class weights...")
    
    # Create more aggressive weights for extreme minority classes
    custom_weights = {0: 10.0, 1: 1.0, 2: 0.8, 3: 8.0, 4: 20.0}
    sample_weights2 = np.array([custom_weights[y] for y in y_sat_train])
    
    weighted_model2 = xgb.XGBClassifier(
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
    
    weighted_model2.fit(X_train, y_sat_train, sample_weight=sample_weights2)
    weighted_pred2 = weighted_model2.predict(X_test)
    
    weighted_accuracy2 = accuracy_score(y_sat_test, weighted_pred2)
    weighted_f1_2 = f1_score(y_sat_test, weighted_pred2, average='weighted')
    weighted_f1_macro2 = f1_score(y_sat_test, weighted_pred2, average='macro')
    
    print(f"Weighted (custom) - Accuracy: {weighted_accuracy2:.4f}")
    print(f"Weighted (custom) - F1 (weighted): {weighted_f1_2:.4f}")
    print(f"Weighted (custom) - F1 (macro): {weighted_f1_macro2:.4f}")
    
    return {
        'balanced': {
            'model': weighted_model1,
            'accuracy': weighted_accuracy1,
            'f1_weighted': weighted_f1_1,
            'f1_macro': weighted_f1_macro1,
            'predictions': weighted_pred1
        },
        'custom': {
            'model': weighted_model2,
            'accuracy': weighted_accuracy2,
            'f1_weighted': weighted_f1_2,
            'f1_macro': weighted_f1_macro2,
            'predictions': weighted_pred2
        }
    }

def detailed_class_analysis(baseline_results, weighted_results, y_test):
    """Detailed per-class analysis"""
    print("\n=== Detailed Per-Class Analysis ===")
    
    models = {
        'Baseline': baseline_results,
        'Weighted (Balanced)': weighted_results['balanced'],
        'Weighted (Custom)': weighted_results['custom']
    }
    
    class_names = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
    
    for model_name, results in models.items():
        print(f"\n{model_name}:")
        report = classification_report(y_test, results['predictions'], output_dict=True)
        
        for i in range(5):
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                support = int(report[str(i)]['support'])
                print(f"  {class_names[i]} (Class {i}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n={support})")
    
    # Find best model based on macro F1 (better for imbalanced data)
    best_model_name = max(models.keys(), key=lambda k: models[k]['f1_macro'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model (by Macro F1): {best_model_name}")
    print(f"  Macro F1-score: {best_model['f1_macro']:.4f}")
    
    return best_model_name, best_model

def save_results(baseline_results, weighted_results, best_model_name, best_model):
    """Save class weight improvement results"""
    print("\nSaving results...")
    
    # Save best model
    joblib.dump(best_model['model'], 'models/class_weighted_satisfaction_model.pkl')
    
    # Prepare results summary
    results_summary = {
        'baseline_accuracy': baseline_results['accuracy'],
        'baseline_f1_weighted': baseline_results['f1_weighted'],
        'baseline_f1_macro': baseline_results['f1_macro'],
        'balanced_weighted_accuracy': weighted_results['balanced']['accuracy'],
        'balanced_weighted_f1_weighted': weighted_results['balanced']['f1_weighted'],
        'balanced_weighted_f1_macro': weighted_results['balanced']['f1_macro'],
        'custom_weighted_accuracy': weighted_results['custom']['accuracy'],
        'custom_weighted_f1_weighted': weighted_results['custom']['f1_weighted'],
        'custom_weighted_f1_macro': weighted_results['custom']['f1_macro'],
        'best_method': best_model_name,
        'best_accuracy': best_model['accuracy'],
        'best_f1_weighted': best_model['f1_weighted'],
        'best_f1_macro': best_model['f1_macro']
    }
    
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv('model_results/class_weight_improvement_results.csv', index=False)
    
    print("Results saved:")
    print("- models/class_weighted_satisfaction_model.pkl")
    print("- model_results/class_weight_improvement_results.csv")
    
    return results_summary

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_sat_train, y_sat_test = load_data()
    
    # Train models
    baseline_results = train_baseline_model(X_train, X_test, y_sat_train, y_sat_test)
    weighted_results = train_weighted_models(X_train, X_test, y_sat_train, y_sat_test)
    
    # Detailed analysis
    best_model_name, best_model = detailed_class_analysis(baseline_results, weighted_results, y_sat_test)
    
    # Save results
    final_results = save_results(baseline_results, weighted_results, best_model_name, best_model)
    
    total_time = time.time() - start_time
    print(f"\n=== Class Weight Improvement Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best method: {best_model_name}")
    print(f"Best Macro F1-score: {final_results['best_f1_macro']:.4f}")
    
    return final_results

if __name__ == "__main__":
    main()