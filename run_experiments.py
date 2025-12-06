import os
import torch
import csv
from train import train_model
from plot_results import plot_experiments

from loguru import logger

# --------------------------------------------------------------------------
# 1. THE BASELINE (Control Group)
# --------------------------------------------------------------------------
# This configuration serves as the "Center Point". 
# Every experiment will be exactly this, with ONE value changed only.
BASELINE_CONFIG = {
    'lr': 1e-3,
    'batch_size': 32,
    'weight_decay': 1e-4,
    'step_size': 7,
    'lambda_reg': 1.0,
    'hidden_dim': 512,
    'dropout': 0.5,
    'backbone': 'resnet18',
    'epochs': 5,      
}

# --------------------------------------------------------------------------
# 2. THE VARIABLES (Experimental Groups)
# --------------------------------------------------------------------------
# We will test these values one by one against the baseline.
param_grid = {
    # --- OPTIMIZATION ---
    'lr': [1e-3, 5e-4, 1e-4],
    'batch_size': [16, 32, 64],
    'weight_decay': [1e-4, 1e-3, 0.0],
    'step_size': [5, 10],

    # --- LOSS BALANCE ---
    'lambda_reg': [1.0, 5.0, 10.0],

    # --- ARCHITECTURE ---
    'hidden_dim': [256, 512, 1024],
    'dropout': [0.3, 0.5],
    'backbone': ['resnet18', 'custom'] 
}

def run_experiments():
    # Setup Directory
    experiment_root = "experiments_ofat" # "One Factor At A Time"
    os.makedirs(experiment_root, exist_ok=True)
    
    # Setup Summary CSV
    # We add 'varied_param' and 'varied_value' columns to track what we changed
    summary_path = os.path.join(experiment_root, "summary.csv")
    csv_headers = ['exp_id', 'varied_param', 'varied_value'] + list(BASELINE_CONFIG.keys()) + ['best_val_mae']
    
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    exp_counter = 1
    total_experiments = sum(len(v) for v in param_grid.values())
    
    logger.info(f"Starting OFAT Experiments. Baseline: {BASELINE_CONFIG}")

    # ----------------------------------------------------------------------
    # MAIN LOOP: Iterate by Parameter
    # ----------------------------------------------------------------------
    for param_name, values_to_test in param_grid.items():
        
        logger.info(f"\n{'='*20} Testing Variable: {param_name} {'='*20}")
        
        for val in values_to_test:
            # 1. Prepare Config: Copy Baseline -> Update ONE value
            config = BASELINE_CONFIG.copy()
            config[param_name] = val
            
            # 2. Create Unique ID
            # e.g. "exp_01_var_lr_0.001"
            exp_id = f"exp_{exp_counter:02d}"
            dir_name = f"{exp_id}_var_{param_name}_{val}"
            save_dir = os.path.join(experiment_root, dir_name)
            
            config['save_dir'] = save_dir
            config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            exp_counter += 1

            # 3. Check if done
            if os.path.exists(os.path.join(save_dir, "metrics.csv")):
                logger.warning(f"Experiment {dir_name} already exists. Skipping.")
                continue

            logger.info(f"Running {exp_id}: Varying '{param_name}' from {BASELINE_CONFIG[param_name]} -> {val}")
            
            # 4. Run Training
            try:
                # Assuming train_model returns the best metric (MAE)
                best_mae = train_model(config)
                
                # 5. Log to Summary CSV
                with open(summary_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Construct row: Metadata + Config Values + Result
                    row = [exp_id, param_name, val] + list(config.values()) + [best_mae]
                    writer.writerow(row)
                    
            except Exception as e:
                logger.error(f"Experiment {dir_name} Failed: {e}")


    logger.info("\nAll Experiments Completed.")
    
    # Generate Plots
    logger.info("Generating Plots...")
    try:
        plot_experiments(experiment_root)
    except Exception as e:
        logger.warning(f"Plotting failed (expected if plot script not updated): {e}")

if __name__ == "__main__":
    run_experiments()