import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load datasets
eeg_data = pd.read_csv('/Users/argyro/BiLSTM-EEG/EEG_data.csv')  # Replace with your EEG data file path
demo_data = pd.read_csv('/Users/argyro/BiLSTM-EEG/demographic_info.csv')  # Replace with demographic info file path
print(demo_data.columns)

# Merge datasets
merged_data = pd.merge(eeg_data, demo_data, left_on='SubjectID', right_on='SubjectID')

# Preprocess features
# EEG Columns
eeg_features = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']

# Normalize data
scaler = StandardScaler()
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

# Print data shape
print(f"Original data shape: {X_padded.shape}")
print(f"Labels shape: {y.shape}")

# BiLSTM Model Definition
class BiLSTMModel(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=False))
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Training function
def train_step(model, inputs, targets, optimizer):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        targets = tf.reshape(targets, (-1,1))
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(targets, logits))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
accuracies = []

for train_index, test_index in loo.split(X_padded):
    X_train, X_test = X_padded[train_index], X_padded[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Print the size of training and test sets
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # Initialize model and optimizer
    model = BiLSTMModel(hidden_size=64, output_size=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        loss = train_step(model, X_train, y_train, optimizer)
        print(f"Epoch {epoch}, Train Loss: {loss:.4f}")

    # Evaluation
    logits = model(X_test)
    predictions = (tf.squeeze(logits).numpy() > 0.5).astype(int)
    predictions = np.array([predictions]) if predictions.ndim == 0 else predictions

    # Print true labels and predictions for debugging
    print(f"True labels: {y_test}")
    print(f"Predictions: {predictions}")

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Fold accuracy: {accuracy:.2f}")
    accuracies.append(accuracy)

# Final accuracy
print(f"LOOCV Accuracy: {np.mean(accuracies):.2f}")
