import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from loguru import logger


def plot_experiments(experiment_root="experiments"):
    """
    Scans the experiment_root directory for subfolders containing 'metrics.csv'.
    Generates comparison plots.
    """
    if not os.path.exists(experiment_root):
        logger.error(f"Directory {experiment_root} does not exist.")
        return

    # 1. Load Data
    all_data = []
    subdirs = [d for d in os.listdir(experiment_root) if os.path.isdir(os.path.join(experiment_root, d))]
    
    for d in subdirs:
        csv_path = os.path.join(experiment_root, d, "metrics.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['experiment_id'] = d
                all_data.append(df)
            except Exception as e:
                logger.error(f"Could not read {csv_path}: {e}")

    if not all_data:
        logger.error("No metrics.csv files found to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # 2. Plot MAE Comparison (Validation)
    plt.figure(figsize=(10, 6))
    for exp_id in combined_df['experiment_id'].unique():
        subset = combined_df[combined_df['experiment_id'] == exp_id]
        plt.plot(subset['epoch'], subset['val_mae'], marker='o', label=exp_id)
    
    plt.title("Validation MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (meters)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_root, "comparison_val_mae.png"))
    plt.close()
    
    # 3. Plot Accuracy Comparison
    plt.figure(figsize=(10, 6))
    for exp_id in combined_df['experiment_id'].unique():
        subset = combined_df[combined_df['experiment_id'] == exp_id]
        plt.plot(subset['epoch'], subset['val_acc'], marker='s', linestyle='--', label=exp_id)
        
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_root, "comparison_val_acc.png"))
    plt.close()

    logger.info(f"Plots saved to {experiment_root}/")

if __name__ == "__main__":
    plot_experiments()
