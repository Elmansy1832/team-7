import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle

def build_mlp_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(data_dir='data'):
    X = []
    y = []
    labels = os.listdir(data_dir)
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            if file.endswith('.npy'):
                X.append(np.load(os.path.join(label_dir, file)))
                y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map

# Load the data
X, y, label_map = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define input shape and number of classes
input_shape = (X_train.shape[1],)
num_classes = len(np.unique(y_train))

# Build and train the model
model = build_mlp_model(input_shape, num_classes)
model.summary()

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Save the model weights
weights = model.get_weights()
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)
