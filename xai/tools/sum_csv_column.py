import pandas as pd

# === Input paths ===
csv_file = '/Users/abhilashreddysomigari/Downloads/YOLOv5_Detector/data/qmetric_out/results_per_image.csv'   # your CSV file
column_name = 'AUC_topK'    # the column you want to average

# === Load CSV ===
df = pd.read_csv(csv_file)

# === Calculate sum ===
sum_value = df[column_name].sum()

print(f"Sum of column '{column_name}': {sum_value:.4f}")
