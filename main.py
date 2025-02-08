import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, utils, callbacks
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "C:\Users\kkdha\Downloads/GTZAN"
GENRES = ['blues', 'classical', 'country', 'disco', 
          'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
SR = 22050
DURATION = 30  # Full track duration
SEGMENT_DURATION = 3  # Seconds per segment
SAMPLES_PER_SEGMENT = SR * SEGMENT_DURATION
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = DURATION // SEGMENT_DURATION
NUM_TIME_STEPS = 130  # MFCC time steps after padding/truncating

def load_and_process_data():
    features = []
    labels = []
    
    for genre_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(DATA_PATH, genre)
        for filename in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, filename)
            
            try:
                # Load entire track
                signal, sr = librosa.load(file_path, sr=SR)
                
                # Split into 3-second segments
                for s in range(NUM_SEGMENTS):
                    start = s * SAMPLES_PER_SEGMENT
                    end = start + SAMPLES_PER_SEGMENT
                    
                    # Handle last segment padding
                    if len(signal[start:end]) < SAMPLES_PER_SEGMENT:
                        segment = np.zeros(SAMPLES_PER_SEGMENT)
                        segment[:len(signal[start:end])] = signal[start:end]
                    else:
                        segment = signal[start:end]
                    
                    # Extract MFCC
                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, 
                                              n_mfcc=N_MFCC,
                                              n_fft=N_FFT,
                                              hop_length=HOP_LENGTH)
                    
                    # Adjust time steps
                    if mfcc.shape[1] < NUM_TIME_STEPS:
                        mfcc = np.pad(mfcc, ((0,0), (0, NUM_TIME_STEPS - mfcc.shape[1])))
                    else:
                        mfcc = mfcc[:, :NUM_TIME_STEPS]
                    
                    features.append(mfcc[..., np.newaxis])  # Add channel dimension
                    labels.append(genre_idx)
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return np.array(features), np.array(labels)

# Data Loading and Splitting
X, y = load_and_process_data()

# Convert labels to one-hot encoding
y = utils.to_categorical(y)

# Split into train (70%), validation (15%), and test (15%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

# Normalization
mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
std = np.std(X_train, axis=(0, 2, 3), keepdims=True)

X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

# CNN Model Architecture
model = models.Sequential([
    layers.Input(shape=(N_MFCC, NUM_TIME_STEPS, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(len(GENRES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training with Early Stopping
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=1)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()