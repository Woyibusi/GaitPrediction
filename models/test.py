import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
X_FILE = "../data/X_scaled.npy"
Y_FILE = "../data/y.npy"
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = "best_lstm_model.h5"

# === LOAD DATA ===
print("📂 Loading data...")
X = np.load(X_FILE)
y = np.load(Y_FILE)

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === BUILD LSTM MODEL ===
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
]

# === TRAIN MODEL ===
print("🚀 Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# === LOAD BEST MODEL ===
print(f"\n📦 Loading best model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# === EVALUATE ON VALIDATION SET ===
print("\n🧠 Evaluating on validation set...")
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"🎯 Validation Accuracy: {val_accuracy:.4f}")

# === METRICS: CLASSIFICATION REPORT ===
y_pred = (model.predict(X_val) > 0.5).astype(int)
print("\n🧾 Classification Report:")
print(classification_report(y_val, y_pred, target_names=["Steady", "Unsteady"]))

# === SAVE & DISPLAY CONFUSION MATRIX ===
print("📊 Confusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred: Steady", "Pred: Unsteady"],
            yticklabels=["True: Steady", "True: Unsteady"])
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # ← 💾 Save to file
plt.show()

# === SAVE & DISPLAY TRAINING CURVES ===
try:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy Over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend()
    plt.title("Loss Over Epochs")

    plt.tight_layout()
    plt.savefig("training_curves.png")  # ← 💾 Save to file
    plt.show()

except Exception as e:
    print(f"⚠️ Could not plot training history: {e}")