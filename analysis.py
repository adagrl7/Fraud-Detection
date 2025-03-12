import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import joblib
import os
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# Print versions for verification
print(f"NumPy version: {np.__version__}")
#print(f"scikit-learn version: {sklearn.__version__}")

def load_and_explore_data(file_path):
    """Load and perform initial exploratory data analysis."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic dataset information
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print("\nSample data:")
    print(df.head())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Class distribution
    fraud_count = df['Class'].sum()
    total_count = len(df)
    print(f"\nFraud transactions: {fraud_count} ({(fraud_count/total_count)*100:.4f}%)")
    print(f"Normal transactions: {total_count - fraud_count} ({((total_count-fraud_count)/total_count)*100:.4f}%)")
    
    return df

def visualize_data(df):
    """Create visualizations for the dataset."""
    # Class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df)
    plt.title("Fraud vs. Non-Fraud Transactions")
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Time vs Amount with fraud highlighted
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Time'], df['Amount'], c=df['Class'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Class (1=Fraud)')
    plt.title('Transaction Time vs Amount')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amount')
    plt.savefig('time_vs_amount.png')
    plt.close()
    
    # Feature distributions
    if 'V1' in df.columns and 'V2' in df.columns:  # Check if V features exist
        # PCA for visualization if many features
        features = [col for col in df.columns if col.startswith('V')]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[features])
        
        plt.figure(figsize=(12, 8))
        plt.scatter(pca_result[df['Class']==0, 0], pca_result[df['Class']==0, 1], 
                   label='Normal', alpha=0.3, c='blue')
        plt.scatter(pca_result[df['Class']==1, 0], pca_result[df['Class']==1, 1], 
                   label='Fraud', alpha=0.8, c='red')
        plt.title('PCA of Transaction Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.savefig('pca_visualization.png')
        plt.close()

def preprocess_data(df, smote_ratio=0.3):
    """Preprocess the dataset: scale features and handle imbalance."""
    # Separate features and target
    if 'Time' in df.columns:
        # Optional: Create time-based features instead of using raw time
        df['Hour'] = df['Time'] // 3600  # Convert to hours
        X = df.drop(columns=['Class', 'Time'])
    else:
        X = df.drop(columns=['Class'])
    
    y = df['Class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')
    
    # Split data first before applying SMOTE (to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Apply SMOTE only to the training data
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set shape: {y_train.value_counts()}")
    print(f"Resampled training set shape: {pd.Series(y_train_resampled).value_counts()}")
    
    return X_train_resampled, X_test, y_train_resampled, y_test, X_scaled, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    results = {}
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced', 
        random_state=RANDOM_SEED,
        n_jobs=-1  # Use all available cores
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'predictions': y_pred_rf,
        'probabilities': y_prob_rf,
        'model': rf_model
    }
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=len(y_train) - sum(y_train) / sum(y_train),  # Handle imbalance
        eval_metric='auc',
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'predictions': y_pred_xgb,
        'probabilities': y_prob_xgb,
        'model': xgb_model
    }
    
    # Neural Network with Early Stopping
    print("\nTraining Neural Network...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    # Use a validation split from the training data
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    
    nn_model.fit(
        X_train_nn, y_train_nn,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    y_prob_nn = nn_model.predict(X_test).flatten()
    y_pred_nn = (y_prob_nn > 0.5).astype(int)
    results['Neural Network'] = {
        'predictions': y_pred_nn,
        'probabilities': y_prob_nn,
        'model': nn_model
    }
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    #joblib.dump(scaler, 'scaler.pkl')
    nn_model.save('models/neural_network_model.keras')
    
    return results

def evaluate_and_compare_models(results, y_test):
    """Evaluate and compare model performance."""
    for model_name, data in results.items():
        y_pred = data['predictions']
        y_prob = data['probabilities']
        
        print(f"\nModel: {model_name}")
        print(classification_report(y_test, y_pred))
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Precision-Recall AUC (better for imbalanced datasets)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print(f"PR AUC Score: {pr_auc:.4f}")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()
        
        # Feature Importance (for tree-based models)
        if model_name in ['Random Forest', 'XGBoost']:
            model = data['model']
            feature_importance = pd.DataFrame()
            if model_name == 'Random Forest':
                feature_importance['importance'] = model.feature_importances_
                feature_importance['feature'] = range(X_train.shape[1])
            else:  # XGBoost
                feature_importance['importance'] = model.feature_importances_
                feature_importance['feature'] = range(X_train.shape[1])
            
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
            plt.title(f'Top 15 Features - {model_name}')
            plt.tight_layout()
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
            plt.close()
    
    # ROC Curves Comparison
    plt.figure(figsize=(10, 8))
    for model_name, data in results.items():
        y_prob = data['probabilities']
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_prob):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig('roc_curve_comparison.png')
    plt.close()

def run_anomaly_detection(X_scaled, y):
    """Run anomaly detection models."""
    print("\nRunning Anomaly Detection Models...")
    
    # Isolation Forest
    print("\nTraining Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,  # Adjust based on expected fraud rate
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    anomaly_scores_iso = iso_forest.decision_function(X_scaled)
    anomaly_preds_iso = iso_forest.predict(X_scaled)
    anomaly_preds_iso = np.where(anomaly_preds_iso == -1, 1, 0)  # Convert to 1 (fraud) and 0 (normal)
    
    # One-Class SVM
    print("\nTraining One-Class SVM...")
    # Use a smaller sample for SVM if dataset is large
    sample_size = min(10000, len(X_scaled))
    indices = np.random.choice(range(len(X_scaled)), sample_size, replace=False)
    
    oc_svm = OneClassSVM(
        nu=0.01,
        kernel='rbf',
        gamma='scale'
    )
    oc_svm.fit(X_scaled[indices])
    anomaly_preds_svm = oc_svm.predict(X_scaled)
    anomaly_preds_svm = np.where(anomaly_preds_svm == -1, 1, 0)
    
    # Local Outlier Factor
    print("\nRunning Local Outlier Factor...")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01,
        n_jobs=-1
    )
    anomaly_preds_lof = lof.fit_predict(X_scaled)
    anomaly_preds_lof = np.where(anomaly_preds_lof == -1, 1, 0)
    
    # Evaluate anomaly detection models
    print("\nIsolation Forest Results:")
    print(classification_report(y, anomaly_preds_iso))
    print(f"ROC AUC: {roc_auc_score(y, -anomaly_scores_iso):.4f}")
    
    print("\nOne-Class SVM Results:")
    print(classification_report(y, anomaly_preds_svm))
    
    print("\nLocal Outlier Factor Results:")
    print(classification_report(y, anomaly_preds_lof))
    
    # Visualize anomaly scores
    plt.figure(figsize=(12, 6))
# Calculate threshold based on percentile of anomaly scores
    threshold = np.percentile(-anomaly_scores_iso, 99)  # Using 99th percentile (matching 1% contamination)
    plt.axvline(x=threshold, color='r', linestyle='--', 
                label="Isolation Forest Threshold (99th percentile)")
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig('anomaly_score_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Load and explore data
    df = load_and_explore_data("data.csv")
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X_scaled, y = preprocess_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Evaluate and compare models
    evaluate_and_compare_models(results, y_test)
    
    # Run anomaly detection
    run_anomaly_detection(X_scaled, y)
    
    print("\nAnalysis complete. Results and visualizations have been saved.")