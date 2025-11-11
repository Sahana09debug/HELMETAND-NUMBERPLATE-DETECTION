from ultralytics.utils.plotting import plot_results  # Correct import
from pathlib import Path

# Path to your training results folder (where 'results.csv' is saved)
results_dir = Path("runs/detect/train")  # Change if your folder name is different

# Plot training graphs (losses, precision, recall, mAP, etc.)
plot_results(results_dir)
print(f"✅ Training graphs saved in: {results_dir}")
