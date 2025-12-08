import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================================
#  1. CONFIGURATION
# ==========================================================
DATASET_FOLDER = r"./cicids2017_csvs/"   # <-- your input folder
OUTPUT_FOLDER = r"./processed/"  # <-- output folder to save merged CSV

# Set True for binary classification (Benign vs Malicious)
BINARY_CLASSIFICATION = False

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
#  2. LABEL MAPPING FUNCTIONS
# ==========================================================


def map_to_5_classes(label):
    label = str(label).lower()

    if "benign" in label:
        return "Benign"
    elif "dns" in label and "mal" not in label:
        return "Suspicious DNS"
    elif "botnet" in label or "ddos" in label:
        return "DDoS/Botnet"
    elif "malware" in label:
        return "Malware"
    elif "phishing" in label:
        return "Phishing"
    else:
        return "Other"


def map_binary(label):
    label = str(label).lower()
    if "benign" in label:
        return "Benign"
    else:
        return "Malicious"


# ==========================================================
#  3. MERGE CSV FILES
# ==========================================================
print(f"📌 Scanning for CSV files in: {DATASET_FOLDER}")

all_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

if len(all_files) == 0:
    raise RuntimeError("❌ No CSV files found in the dataset folder.")

df_list = []
for file in all_files:
    path = os.path.join(DATASET_FOLDER, file)
    print("➡ Merging:", file)

    try:
        df_list.append(pd.read_csv(path))
    except Exception as e:
        print("❌ Error reading", file, ":", e)

data = pd.concat(df_list, ignore_index=True)
print("✔ Merge complete. Shape:", data.shape)

# ==========================================================
#  4. SAVE RAW MERGED CSV
# ==========================================================
raw_output_path = os.path.join(OUTPUT_FOLDER, "merged_raw.csv")
data.to_csv(raw_output_path, index=False)
print(f"📁 Raw merged CSV saved to:\n{raw_output_path}")

# ==========================================================
#  5. DETECT LABEL COLUMN
# ==========================================================
possible_labels = [
    'Label', 'label', 'Attack', 'attack',
    'Attack_type', 'Class', 'class',
    'Malicious', 'malicious'
]

label_col = None
for col in data.columns:
    if col in possible_labels:
        label_col = col
        break

if label_col is None:
    raise ValueError("❌ No valid label column found in dataset.")

print(f"✔ Label column detected: {label_col}")

data = data[data[label_col].notna()]

# ==========================================================
#  6. APPLY LABEL MAPPING
# ==========================================================
print("🏷 Mapping labels...")

if BINARY_CLASSIFICATION:
    data[label_col] = data[label_col].apply(map_binary)
else:
    data[label_col] = data[label_col].apply(map_to_5_classes)

print("✔ Label mapping complete!")
print(data[label_col].value_counts())

# ==========================================================
#  7. CLEAN DATA
# ==========================================================
print("🧹 Cleaning non-numeric columns...")

data = data.loc[:, data.apply(
    lambda col: pd.api.types.is_numeric_dtype(col) or col.name == label_col)]
data = data.dropna()

clean_output_path = os.path.join(OUTPUT_FOLDER, "merged_cleaned.csv")
data.to_csv(clean_output_path, index=False)

print(f"📁 Cleaned merged CSV saved to:\n{clean_output_path}")
print("✔ Cleaned dataset shape:", data.shape)

# ==========================================================
#  8. PREPARE FEATURES
# ==========================================================
X = data.drop(columns=[label_col])
y = data[label_col]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================================
#  9. TRAIN MODEL
# ==========================================================
print("🚀 Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("✔ Model training complete!")

# ==========================================================
#  10. EVALUATION
# ==========================================================
y_pred = model.predict(X_test)

print("\n🎯 ACCURACY:", accuracy_score(y_test, y_pred))

print("\n📘 CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("\n✅ SCRIPT FINISHED SUCCESSFULLY.")
