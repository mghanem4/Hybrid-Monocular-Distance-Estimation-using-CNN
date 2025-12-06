import os
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
def plot_experiments(experiment_root="experiments_ofat"):
    """
    Scans the experiment_root directory.
    Groups experiments by the parameter they varied (e.g., 'lr', 'batch_size').
    Generates specific comparison plots for each parameter.
    """
    if not os.path.exists(experiment_root):
        logger.error(f"Directory {experiment_root} does not exist. Check your run_experiments.py output folder.")
        return

    # 1. Load Data
    all_data = []
    subdirs = [d for d in os.listdir(experiment_root) if os.path.isdir(os.path.join(experiment_root, d))]
    
    # We need to group experiments. 
    # Structure: groups = {'lr': [df1, df2], 'batch_size': [df3, df4]}
    groups = {}

    for d in subdirs:
        # Parse the directory name: "exp_01_var_lr_0.001"
        if "_var_" not in d:
            continue # Skip folders that don't match the OFAT pattern
            
        # Split string to find which parameter changed
        # "exp_01" | "lr_0.001"
        parts = d.split("_var_")
        if len(parts) < 2: continue
        
        # This part is tricky (value might contain underscores), so  use known keys
        rest = parts[1] # e.g. "learning_rate_0.001" or "backbone_resnet18"
        
        # Identify the parameter name
        param_name = "unknown"
        clean_val = "unknown"
        
        # List of known params from your grid
        known_params = ['lr', 'batch_size', 'weight_decay', 'step_size', 'lambda_reg', 'hidden_dim', 'dropout', 'backbone']
        
        for k in known_params:
            if rest.startswith(k):
                param_name = k
                # Extract value string (remove param name + underscore)
                clean_val = rest[len(k)+1:] 
                break
        
        # Load CSV
        csv_path = os.path.join(experiment_root, d, "metrics.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['label'] = f"{param_name}={clean_val}" # Label for the legend
                
                if param_name not in groups:
                    groups[param_name] = []
                groups[param_name].append(df)
                
            except Exception as e:
                logger.error(f"Could not read {csv_path}: {e}")

    if not groups:
        logger.error("No valid OFAT experiments found to plot.")
        return

    # 2. Generate One Plot Per Parameter
    output_dir = os.path.join(experiment_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for param_name, df_list in groups.items():
        logger.info(f"Generating plot for: {param_name} ({len(df_list)} lines)")
        
        # --- Plot MAE ---
        plt.figure(figsize=(10, 6))
        for df in df_list:
            plt.plot(df['epoch'], df['val_mae'], marker='.', label=df['label'].iloc[0])
        
        plt.title(f"Effect of '{param_name}' on Validation Error (MAE)")
        plt.xlabel("Epoch")
        plt.ylabel("MAE (meters)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"effect_of_{param_name}_mae.png"))
        plt.close()

        # --- Plot Accuracy ---
        plt.figure(figsize=(10, 6))
        for df in df_list:
            plt.plot(df['epoch'], df['val_acc'], marker='.', linestyle='--', label=df['label'].iloc[0])
            
        plt.title(f"Effect of '{param_name}' on Classification Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"effect_of_{param_name}_acc.png"))
        plt.close()

    logger.info(f"All plots saved to {output_dir}/")

def plot_trained_metrics(fp, save_dir= "."):
    data = pd.read_csv(fp)
    # --- PART 1: Plot metrics One by One ---
    # 1. Training Loss
    plt.figure(figsize=(8, 6))
    plt.plot(data['epoch'], data['train_loss'], color='#e74c3c', linewidth=2, marker='o', label='Train Loss')
    plt.title('Training Loss', fontsize=14)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss_plot.png"))

    # 2. Error Metrics
    plt.figure(figsize=(8, 6))
    plt.plot(data['epoch'], data['train_mae'], label='Train MAE', linestyle='--', color='#3498db')
    plt.plot(data['epoch'], data['val_mae'], label='MAE', color='#2980b9')
    plt.plot(data['epoch'], data['val_rmse'], label='RMSE', color='#8e44ad')
    plt.title('Error Metrics (MAE & RMSE)', fontsize=14)
    plt.ylabel('Error Value')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_metrics.png"))

    # 3. Performance Scores
    plt.figure(figsize=(8, 6))
    plt.plot(data['epoch'], data['val_acc'], label='Acc', color='#2ecc71')
    plt.plot(data['epoch'], data['val_nearest_acc'], label='Nearest Acc', linestyle='--', color='#27ae60')
    plt.plot(data['epoch'], data['val_r2'], label='R2 Score', color='#f1c40f')
    plt.title('Performance Scores (Accuracy & R2)', fontsize=14)
    plt.ylabel('Score (0-1)')
    plt.xlabel('Epoch')
    plt.ylim(0.9, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "perform_score.png"))

    # --- PART 2: Plot All Together ---
    
    # Create a figure with 3 subplots side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Training Loss (Lowest scale)
    ax1.plot(data['epoch'], data['train_loss'], color='#e74c3c', linewidth=2, marker='o', label='Train Loss')
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Error Metrics (MAE & RMSE)
    # We group these because they share similar scales (approx 1.0 to 4.0)
    ax2.plot(data['epoch'], data['train_mae'], label='Train MAE', linestyle='--', color='#3498db')
    ax2.plot(data['epoch'], data['val_mae'], label='MAE', color='#2980b9')
    ax2.plot(data['epoch'], data['val_rmse'], label='RMSE', color='#8e44ad')
    ax2.set_title('Error Metrics (MAE & RMSE)', fontsize=14)
    ax2.set_ylabel('Error Value')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Plot 3: Accuracy and R2 Scores
    # We group these because they are all normalized between 0 and 1
    ax3.plot(data['epoch'], data['val_acc'], label='Acc', color='#2ecc71')
    ax3.plot(data['epoch'], data['val_nearest_acc'], label='Nearest Acc', linestyle='--', color='#27ae60')
    ax3.plot(data['epoch'], data['val_r2'], label='R2 Score', color='#f1c40f')
    ax3.set_title('Performance Scores (Accuracy & R2)', fontsize=14)
    ax3.set_ylabel('Score (0-1)')
    ax3.set_ylim(0.9, 1.0) # Zoom in to see the fine detail since values are high
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='lower right')

    # Common X-axis label
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"combined_metrics.png"))
    plt.show()

if __name__ == "__main__":
    plot_trained_metrics("custom_best_model/metrics.csv")
    # plot_experiments()