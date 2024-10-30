# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from data_preprocessing import connect_db, extract_data, prepare_model_data
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

def build_model(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    # Preprocess Data
    engine = connect_db()
    data = extract_data(engine)
    X, y, label_encoders = prepare_model_data(data)
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    roc_aucs = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Build and Compile Model
        model = build_model(X_train.shape[1])
        model = compile_model(model)
        
        # Train Model
        history = model.fit(
            X_train, 
            y_train, 
            epochs=50, 
            batch_size=8, 
            validation_split=0.2, 
            verbose=0
        )
        
        # Evaluate Model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold} Accuracy: {accuracy * 100:.2f}%")
        accuracies.append(accuracy)
        
        # Predictions for Classification Report and ROC-AUC
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC-AUC Score
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            print(f"Fold {fold} ROC-AUC Score: {roc_auc:.2f}")
            roc_aucs.append(roc_auc)
        else:
            print(f"Fold {fold} ROC-AUC Score is not defined (only one class present).")
    
    # Average Metrics
    print(f"\nAverage Accuracy: {np.mean(accuracies) * 100:.2f}%")
    if roc_aucs:
        print(f"Average ROC-AUC Score: {np.mean(roc_aucs):.2f}")
    else:
        print("ROC-AUC Score not available for any fold.")
    
    # Train Final Model on Entire Dataset
    print("\nTraining final model on the entire dataset...")
    final_model = build_model(X.shape[1])
    final_model = compile_model(final_model)
    final_model.fit(X, y, epochs=50, batch_size=8, verbose=1)
    final_model.save('cdss_model.h5')
    print("Final model saved as 'cdss_model.h5'.")
    
    # Save Label Encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Label encoders saved as 'label_encoders.pkl'.")
    
    print("Model training and saving completed successfully.")

if __name__ == "__main__":
    train_model()

