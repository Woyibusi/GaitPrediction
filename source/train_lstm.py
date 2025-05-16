import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import RandomSearch
import joblib

# === CONFIGURATION ===
X_FILE = "../data/X_scaled.npy"
Y_FILE = "../data/y.npy"
MAX_TRIALS = 30
EPOCHS = 50
PROJECT_NAME = "lstm_tuning"
MODEL_DIR = "../data/tuner_results"

# === LOAD DATA ===
print("📂 Loading data...")
X = np.load(X_FILE)
y = np.load(Y_FILE)

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === DEFINE MODEL BUILD FUNCTION ===
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int("lstm_units", 32, 128, step=32),
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=False
    ))
    model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(Dense(
        units=hp.Int("dense_units", 16, 64, step=16),
        activation='relu'
    ))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log")
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# === SETUP TUNER ===
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=MAX_TRIALS,
    executions_per_trial=1,
    directory=MODEL_DIR,
    project_name=PROJECT_NAME,
    overwrite=True
)

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# === RUN TUNING ===
print("🚀 Starting hyperparameter search...")
tuner.search(X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=EPOCHS,
             batch_size=32,
             callbacks=callbacks,
             verbose=1)

# === RETRIEVE BEST MODEL & PARAMS ===
best_model = tuner.get_best_models(1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

print("\n✅ Best Hyperparameters:")
print(f"LSTM Units: {best_hps.get('lstm_units')}")
print(f"Dense Units: {best_hps.get('dense_units')}")
print(f"Dropout: {best_hps.get('dropout')}")
print(f"Learning Rate: {best_hps.get('lr')}")

# === SAVE BEST MODEL ===
best_model.save("best_lstm_model.h5")
print("✅ Best model saved to: best_lstm_model.h5")

# === OPTIONAL: SAVE HYPERPARAMETERS ===
joblib.dump(best_hps.values, "best_hyperparameters.pkl")
print("📄 Best hyperparameters saved to: best_hyperparameters.pkl")
