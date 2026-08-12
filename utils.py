# utils.py
# Helper functions for saving and managing results

import pandas as pd
import json  
import os
from datetime import datetime

def save_results(results, filename='experiment_results.csv'):
    """Save results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df

def load_results(filename='experiment_results.csv'):
    """Load results from CSV file"""
    return pd.read_csv(filename)

def save_experiment_config(config, model_config, results, filename='experiment_log.json'):
    """Save complete configuration and results of an experiment"""
    log = {
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'model_config': model_config,
        'results': {
            'test_acc': results['test_acc'],
            'test_loss': results['test_loss'],
            'train_time': results['train_time'],
            'history': results['history']
        }
    }
    with open(filename, 'w') as f:
        json.dump(log, f, indent=2)

def format_time(seconds):
    """Convert seconds to readable format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)