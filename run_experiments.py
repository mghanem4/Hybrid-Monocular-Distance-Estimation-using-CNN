import os
import torch
import csv
import itertools
from train import train_model
from plot_results import plot_experiments

from loguru import logger

# Define Hyperparameter Grid
param_grid = {
    'lr': [0.001, 0.0005],
    'backbone': ['custom', 'resnet18'],
    'batch_size': [32], 
}

def run_experiments():
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    experiment_root = "experiments"
    os.makedirs(experiment_root, exist_ok=True)
    
    # Summary CSV
    summary_path = os.path.join(experiment_root, "summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['exp_id', 'lr', 'backbone', 'batch_size', 'best_val_mae'])
    
    for i, combo in enumerate(combinations):
        # Create config dict
        params = dict(zip(keys, combo))
        exp_id = f"exp_{i+1:02d}"
        
        # Construct directory name
        dir_name = f"{exp_id}_lr{params['lr']}_{params['backbone']}"
        save_dir = os.path.join(experiment_root, dir_name)
        
        # Skip if already done (simple check)
        if os.path.exists(os.path.join(save_dir, "metrics.csv")):
            logger.warning(f"Experiment {dir_name} already exists. Skipping.")
            continue

        logger.info(f"\n{'#'*40}")
        logger.info(f"Running Experiment {i+1}/{len(combinations)}: {dir_name}")
        logger.info(f"Params: {params}")
        logger.info(f"{'#'*40}")
        
        # Build Config
        config = {
            'batch_size': params['batch_size'],
            'lr': params['lr'],
            'epochs': 5, # Kept small for demo; increase for real runs
            'weight_decay': 1e-4,
            'lambda_reg': 1.0,
            'backbone': params['backbone'],
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'save_dir': save_dir
        }
        
        # Run Training
        try:
            best_mae = train_model(config)
            
            # Log to Summary
            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([exp_id, params['lr'], params['backbone'], params['batch_size'], best_mae])
                
        except Exception as e:
            logger.error(f"Experiment {exp_id} Failed: {e}")

    logger.info("\nAll Experiments Completed.")
    
    # Generate Plots automatically
    logger.info("Generating Plots...")
    plot_experiments(experiment_root)

if __name__ == "__main__":
    run_experiments()
