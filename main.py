import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load datasets
eeg_data = pd.read_csv('path_to_eeg_data.csv')  # Replace with your EEG data file path
demo_data = pd.read_csv('path_to_demographics.csv')  # Replace with demographic info file path

# Merge datasets
merged_data = pd.merge(eeg_data, demo_data, left_on='SubjectID', right_on='Subject ID')

# Preprocess features
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

# Define JAX/Flax Model
class BiLSTMModel(nn.Module):
    hidden_size: int
    output_size: int

    def setup(self):
        self.lstm = nn.LSTMCell(name="lstm_cell")
        self.fc1 = nn.Dense(32)
        self.fc2 = nn.Dense(self.output_size)

    def __call__(self, x):
        # LSTM
        state = self.lstm.initialize_carry(rng=jax.random.PRNGKey(0), batch_size=x.shape[0], input_size=1)
        outputs = []
        for t in range(x.shape[1]):  # Iterate over time steps
            state, out = self.lstm(state, x[:, t])
            outputs.append(out)
        
        x = jnp.stack(outputs, axis=1)  # Stack the LSTM outputs
        x = x[:, -1, :]  # Use the last time step output
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        x = jax.nn.sigmoid(x)
        return x

# Create the training state
def create_train_state(model, learning_rate):
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, max_sequence_length, X_padded.shape[2])))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
def train_step(state, batch):
    def loss_fn(params):
        inputs, targets = batch
        logits = state.apply_fn({'params': params}, inputs)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
accuracies = []

for train_index, test_index in loo.split(X_padded):
    X_train, X_test = X_padded[train_index], X_padded[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create DataLoader (custom function)
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    # Initialize model and optimizer
    model = BiLSTMModel(hidden_size=64, output_size=1)
    state = create_train_state(model, learning_rate=0.001)

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        state, train_loss = train_step(state, train_data)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

    # Evaluation
    logits = state.apply_fn({'params': state.params}, X_test)
    predictions = (logits.squeeze() > 0.5).astype(int)  # Binary classification
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

# Final accuracy
print(f"LOOCV Accuracy: {np.mean(accuracies):.2f}")
