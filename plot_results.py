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
        
        # This part is tricky (value might contain underscores), so we use known keys
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

if __name__ == "__main__":
    plot_experiments()