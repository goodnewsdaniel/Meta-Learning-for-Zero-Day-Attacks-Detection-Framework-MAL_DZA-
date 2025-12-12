import pandas as pd

# Load both files
train = pd.read_csv(
    "./dataset_bank/unsw_nb15_merged/UNSW_NB15_training-set.csv")
test = pd.read_csv("./dataset_bank/unsw_nb15_merged/UNSW_NB15_testing-set.csv")

# Verify the label column names
print(train.columns)
print(test.columns)

# Ensure both files contain the same columns in same order
assert list(train.columns) == list(test.columns), "Column mismatch!"

# Merge them properly
merged = pd.concat([train, test], ignore_index=True)

# Check class distribution
print(merged['label'].value_counts())
print(merged['attack_cat'].value_counts())

# Save merged file
merged.to_csv("./dataset_bank/unsw_nb15_merged/UNSW_NB15_FULL.csv", index=False)

print("Merged dataset saved successfully!")
