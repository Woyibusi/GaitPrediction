import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

# === LOAD DATA ===
X = np.load("../data/X_scaled.npy")   # shape: (384, 48, 4)
y = np.load("../data/y.npy")          # shape: (384,)

# === SPLIT (Same split as in training!) ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === LOAD MODEL ===
model = load_model("../models/best_lstm_model.h5")

# === PREDICT ===
y_pred = model.predict(X_val)
y_pred_class = (y_pred > 0.5).astype(int)

# === METRICS ===
print("\n🧾 Classification Report:")
print(classification_report(y_val, y_pred_class, target_names=["Steady", "Unsteady"]))

print("📊 Confusion Matrix:")
cm = confusion_matrix(y_val, y_pred_class)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred: Steady", "Pred: Unsteady"],
            yticklabels=["True: Steady", "True: Unsteady"])
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("👀 Sample predictions:", model.predict(X_val[:5]))
print("🟩 True labels:", y_val[:5])
