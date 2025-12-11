import os
import pandas as pd

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
        # Use low_memory=False to avoid dtype inference issues
        df = pd.read_csv(path, low_memory=False)
        df_list.append(df)
        print(f"   ✔ Loaded {len(df)} rows")
    except Exception as e:
        print("❌ Error reading", file, ":", str(e)[:100])

if not df_list:
    raise RuntimeError("❌ No CSV files could be successfully read.")

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
    # Check exact match first
    if col in possible_labels:
        label_col = col
        break
    # Check case-insensitive match
    if col.strip().lower() in [l.lower() for l in possible_labels]:
        label_col = col
        break

if label_col is None:
    # If still not found, print available columns for debugging
    print("❌ No valid label column found in dataset.")
    print("Available columns:", list(data.columns))
    raise ValueError("Please ensure a label/class column exists in the dataset.")

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

print("\n✅ MERGE AND SAVE COMPLETE.")
