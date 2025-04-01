import splitfolders

# Path to your downloaded dataset
INPUT_FOLDER = r"D:\GitHub\Google-Meet-Sign-Language-Convertor\backend\data"  

# Path to save the split dataset
OUTPUT_FOLDER = r"D:\GitHub\Google-Meet-Sign-Language-Convertor\backend\SplitDataSet"

# Split dataset into 80% train, 10% validation, 10% test
splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER, seed=42, ratio=(0.8, 0.1, 0.1))

print(f"Dataset split successfully! New dataset is in: {OUTPUT_FOLDER}")
