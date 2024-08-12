# src/train_model.py
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load the data
df_patients = pd.read_csv('data/simpledata.csv')
df_medications = pd.read_csv('data/simplemedications.csv')

# Merge datasets on PatientID
df = pd.merge(df_patients, df_medications, on='PatientID')

# Convert categorical data to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Medication'] = df['Medication'].map({'MedA': 0, 'MedB': 1, 'MedC': 2})

# Features and target
X = df[['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'Diabetes', 'Medication', 'Dosage']]
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

# Save the model
model.save('models/ai_cdss_model.h5')

print("Model trained and saved.")
