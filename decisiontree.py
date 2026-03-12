import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
from sklearn import tree


# ====== Load data ======
try:
    data = pd.read_csv("data_iklim_fuzzy.csv")  # ganti dengan nama file lo
    print("✅ Dataset berhasil dibaca")
except FileNotFoundError:
    raise FileNotFoundError("❌ File data_iklim_fuzzy.csv not found, check your path!")

print("Kolom dataset:", data.columns.tolist())

drop_cols = [col for col in ["date", "label"] if col in data.columns]
X = data.drop(drop_cols, axis=1)  # fitur numerik
y = data["label"] if "label" in data.columns else None

if y is None:
    raise ValueError("❌ Column 'label' not found in the dataset!")

# ====== Split train/test ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== Model Decision Tree ======
model = DecisionTreeClassifier(
    criterion="entropy", 
    max_depth=5, 
    random_state=42
)
model.fit(X_train, y_train)

# ====== Evaluasi ======
yy_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("🔎 Evaluasi Model")
print("Accuracy (Train):", round(accuracy_score(y_train, yy_pred_train), 3))
print("Accuracy (Test) :", round(accuracy_score(y_test, y_pred_test), 3))

# Classification report hanya untuk test (lebih penting)
print("\nClassification Report (Test Data):\n", classification_report(y_test, y_pred_test))

# Mapping label prediksi test (opsional buat cek hasil per instance)
label_mapping = {0: "aman", 1: "siaga", 2: "banjir"}
y_pred_labels = [label_mapping[p] for p in y_pred_test]

# ====== Simpan model ======
joblib.dump(
    {"model": model, "feature_names": X.columns.tolist()},
    "decision_tree_banjir.pkl"
)
print("✅ Model disimpan ke decision_tree_banjir.pkl")

# ====== Visualisasi Decision Tree ======
plt.figure(figsize=(16,8))
tree.plot_tree(
    model, 
    feature_names=X.columns, 
    class_names=["aman", "siaga", "banjir"], 
    filled=True, 
    rounded=True
)
plt.title("Visualisasi Decision Tree")
plt.tight_layout()
plt.show()