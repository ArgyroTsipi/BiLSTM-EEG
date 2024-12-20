import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Masking
from tensorflow.keras.utils import to_categorical

# Load datasets
eeg_data = pd.read_csv('path_to_eeg_data.csv')  # Replace with your EEG data file path
demo_data = pd.read_csv('path_to_demographics.csv')  # Replace with demographic info file path

# Merge datasets
merged_data = pd.merge(eeg_data, demo_data, left_on='SubjectID', right_on='Subject ID')

# Preprocess features
# Normalize EEG features
scaler = StandardScaler()
eeg_features = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
merged_data[eeg_features] = scaler.fit_transform(merged_data[eeg_features])

# Encode demographic features
encoder = OneHotEncoder()
demographic_features = pd.DataFrame(encoder.fit_transform(merged_data[['gender', 'ethnicity']]).toarray())
merged_data = pd.concat([merged_data, demographic_features], axis=1)

# Prepare sequences and labels
X = merged_data[eeg_features + demographic_features.columns.tolist()].values
y = merged_data['user-definedlabeln'].values  # Replace with the correct label column
y_binary = np.array(y).astype('int')  # Binary target

# Reshape for time-series (group by SubjectID and VideoID for sequences)
subjects = merged_data['SubjectID'].unique()
sequences = [merged_data[merged_data['SubjectID'] == subj][eeg_features].values for subj in subjects]
labels = [merged_data[merged_data['SubjectID'] == subj]['user-definedlabeln'].values[0] for subj in subjects]

# Pad sequences if needed
max_sequence_length = max([len(seq) for seq in sequences])
X_padded = np.array([np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), mode='constant') for seq in sequences])
y = np.array(labels)

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
accuracies = []

for train_index, test_index in loo.split(X_padded):
    X_train, X_test = X_padded[train_index], X_padded[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Build the BiLSTM model
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_sequence_length, len(eeg_features))))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)
    
    # Evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(acc)

# Final accuracy
print(f"LOOCV Accuracy: {np.mean(accuracies):.2f}")
