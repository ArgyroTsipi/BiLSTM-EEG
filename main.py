import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Prepare sequences and labels (group by SubjectID and VideoID)
sequences = []
labels = []

# Group by SubjectID and VideoID
for (subject, video), group in merged_data.groupby(['SubjectID', 'VideoID']):
    sequences.append(group[eeg_features].values)
    labels.append(group['user-definedlabeln'].values[0])  # Binary confusion label for the video

# Pad sequences to match the longest video sequence
max_sequence_length = max([len(seq) for seq in sequences])
X_padded = np.array([np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), mode='constant') for seq in sequences])
y = np.array(labels)

# Print data shape
print(f"Padded data shape: {X_padded.shape}")
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
precisions = []
recalls = []
f1_scores = []

for train_index, test_index in loo.split(X_padded):
    X_train, X_test = X_padded[train_index], X_padded[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Print the size of training and test sets
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # Initialize model and optimizer
    model = BiLSTMModel(hidden_size=64, output_size=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Track training loss
    training_losses = []
    training_losses_all_folds = [] 

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        loss = train_step(model, X_train, y_train, optimizer)
        training_losses.append(loss.numpy())  # Append loss to the list
        print(f"Epoch {epoch}, Train Loss: {loss:.4f}")
        
    training_losses_all_folds.append(training_losses)
    
  # Plot the training loss after all epochs for each fold
    #plt.figure(figsize=(10, 6))
    #plt.plot(range(1, 11), training_losses, marker='o', label='Training Loss')
    #plt.title(f'Training Loss per Epoch - Fold {len(accuracies)+1}')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.grid()
    #plt.show()
    #plt.ion() - del
    #plt.savefig() - del
    # Evaluation
    logits = model(X_test)
    predictions = (tf.squeeze(logits).numpy() > 0.5).astype(int)
    predictions = np.array([predictions]) if predictions.ndim == 0 else predictions

    # Print true labels and predictions for debugging
    print(f"True labels: {y_test}")
    print(f"Predictions: {predictions}")

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=1)
    recall = recall_score(y_test, predictions, zero_division=1)
    f1 = f1_score(y_test, predictions, zero_division=1)

    # Log metrics
    print(f"Fold Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Final Metrics
print(f"LOOCV Accuracy: {np.mean(accuracies):.2f}")
print(f"LOOCV Precision: {np.mean(precisions):.2f}")
print(f"LOOCV Recall: {np.mean(recalls):.2f}")
print(f"LOOCV F1-Score: {np.mean(f1_scores):.2f}")

###########################################################
# Create a range for folds
folds = list(range(1, len(accuracies) + 1))

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(folds, accuracies, marker='o', label='Accuracy')
plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean Accuracy ({np.mean(accuracies):.2f})')
plt.title('Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('/Users/argyro/BiLSTM-EEG/plots/accuracy.png')

# Plot Precision
plt.figure(figsize=(10, 6))
plt.plot(folds, precisions, marker='o', label='Precision', color='orange')
plt.axhline(y=np.mean(precisions), color='r', linestyle='--', label=f'Mean Precision ({np.mean(precisions):.2f})')
plt.title('Precision per Fold')
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('/Users/argyro/BiLSTM-EEG/plots/precision.png')

# Plot Recall
plt.figure(figsize=(10, 6))
plt.plot(folds, recalls, marker='o', label='Recall', color='green')
plt.axhline(y=np.mean(recalls), color='r', linestyle='--', label=f'Mean Recall ({np.mean(recalls):.2f})')
plt.title('Recall per Fold')
plt.xlabel('Fold')
plt.ylabel('Recall')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('/Users/argyro/BiLSTM-EEG/plots/recall.png')

# Plot F1-Score
plt.figure(figsize=(10, 6))
plt.plot(folds, f1_scores, marker='o', label='F1-Score', color='purple')
plt.axhline(y=np.mean(f1_scores), color='r', linestyle='--', label=f'Mean F1-Score ({np.mean(f1_scores):.2f})')
plt.title('F1-Score per Fold')
plt.xlabel('Fold')
plt.ylabel('F1-Score')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('/Users/argyro/BiLSTM-EEG/plots/f1score.png')

# Create a subplot with histograms and box plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Histograms on the left
axs[0].hist(accuracies, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axs[0].axvline(np.mean(accuracies), color='r', linestyle='--', label=f'Mean Accuracy ({np.mean(accuracies):.2f})')
axs[0].set_title('Accuracy Distribution across Folds')
axs[0].set_xlabel('Accuracy')
axs[0].set_ylabel('Frequency')
axs[0].legend()
axs[0].grid(True)

# Box plot on the right
axs[1].boxplot([accuracies, precisions, recalls, f1_scores], 
               labels=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
               patch_artist=True, 
               boxprops=dict(facecolor="skyblue", color="black"), 
               flierprops=dict(markerfacecolor='r', marker='o', markersize=8))
axs[1].set_title('Boxplot of Metrics')
axs[1].set_ylabel('Metric Value')
axs[1].grid(True)

plt.tight_layout()
#plt.show()
plt.savefig('/Users/argyro/BiLSTM-EEG/plots/histobox.png')
