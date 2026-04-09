import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ✅ use tensorflow.keras

# ── Config ──
MODEL_PATH  = "skin_major_model.keras"
TEST_DIR    = "dataset/test"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32

LABELS = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Eczema",
    "Lupus",
    "SkinCancer",
    "Vasculitis",
    "Warts"
]

# ── Load model ──
print("🔄 Loading model...")
model = load_model(MODEL_PATH)

# ── Load test data ──
print("📂 Loading test images...")
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False              # ← CRITICAL: never shuffle for evaluation
)

# ── Predictions ──
print("🤖 Running predictions on test set...")
preds  = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# ── Confusion Matrix ──
print("📊 Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS,
    linewidths=0.5
)
plt.title("Confusion Matrix — Skin Disease CNN", fontsize=16, pad=20)
plt.ylabel("Actual Label",    fontsize=13)
plt.xlabel("Predicted Label", fontsize=13)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0,  fontsize=10)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.savefig("static/confusion_matrix.png", dpi=150)
plt.show()
print("✅ Saved: confusion_matrix.png")

# ── Report ──
print("\n===== Classification Report =====")
print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc * 100:.2f}%")

print("\n===== Per-Class Accuracy =====")
for i, label in enumerate(LABELS):
    mask    = (y_true == i)
    correct = (y_pred[mask] == i).sum()
    total   = mask.sum()
    print(f"  {label:<22} {correct:>4}/{total:<4}  ({correct/total*100:.1f}%)")