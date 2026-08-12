# experiments.py
# Systematic execution of all experiments - FIXED VERSION

from config import Config
from trainer_fixed import run_experiment
from utils import save_results, create_directory
import pandas as pd
import time
import os
import traceback

def run_all_experiments():
    """Run all planned experiments systematically"""
    
    # Base configuration
    config = Config()
    create_directory('results')
    
    print("="*70)
    print("🧠 NEURAL NETWORK PARAMETER ANALYSIS")
    print("="*70)
    print(f"📊 Dataset: {config.DATASET}")
    print(f"📊 Epochs per experiment: {config.EPOCHS}")
    print(f"📊 Total experiments planned: 34")
    print("="*70)
    
    # ============ Experiment Definitions ============
    
    # 1. Optimizer experiments (3 types)
    optimizers = ['adam', 'sgd', 'adamw']
    
    # 2. Layer experiments (3 configurations)
    layer_configs = [
        {'name': 'Shallow', 'hidden_sizes': [128]},
        {'name': 'Medium', 'hidden_sizes': [256, 128]},
        {'name': 'Deep', 'hidden_sizes': [256, 128, 64]}
    ]
    
    # 3. BatchNorm experiments (2 states)
    batchnorm_options = [True, False]
    
    # 4. Dropout experiments (3 rates)
    dropout_rates = [0.0, 0.3, 0.5]
    
    # 5. Hyperparameter Tuning (10 random combinations)
    hp_tuning = [
        {'lr': 0.01, 'batch_size': 64},
        {'lr': 0.001, 'batch_size': 128},
        {'lr': 0.0001, 'batch_size': 256},
        {'lr': 0.01, 'batch_size': 128, 'weight_decay': 0.001},
        {'lr': 0.001, 'batch_size': 64, 'weight_decay': 0.0001},
        {'lr': 0.0001, 'batch_size': 256, 'weight_decay': 0.01},
        {'lr': 0.01, 'batch_size': 256, 'weight_decay': 0.0001},
        {'lr': 0.001, 'batch_size': 128, 'weight_decay': 0.01},
        {'lr': 0.0001, 'batch_size': 64, 'weight_decay': 0.001},
        {'lr': 0.005, 'batch_size': 128, 'weight_decay': 0.0005}
    ]
    
    # ============ Results Storage ============
    
    all_results = []
    experiment_counter = 0
    failed_experiments = 0
    start_time_total = time.time()
    
    # ============ Section 1: Optimizer Analysis ============
    
    print("\n" + "="*70)
    print("🔍 SECTION 1: OPTIMIZER ANALYSIS")
    print("="*70)
    print("Testing: Adam, SGD, AdamW")
    print("Network depths: Shallow, Medium, Deep")
    print("-"*70)
    
    for opt in optimizers:
        for layer in layer_configs:
            model_config = {
                'hidden_sizes': layer['hidden_sizes'],
                'use_batchnorm': False,
                'dropout_rate': 0.0
            }
            
            # Update optimizer in config
            config.OPTIMIZER = opt
            
            experiment_counter += 1
            print(f"\n🔬 Experiment {experiment_counter}/34: {opt.upper()} + {layer['name']}")
            print(f"   Config: hidden_sizes={layer['hidden_sizes']}, BN=False, Dropout=0.0")
            
            try:
                result = run_experiment(config, model_config)
                
                # Ensure result has all required fields
                if result and 'test_acc' in result:
                    all_results.append({
                        'experiment_id': experiment_counter,
                        'category': 'optimizer',
                        'optimizer': opt,
                        'layers': layer['name'],
                        'hidden_sizes': str(layer['hidden_sizes']),
                        'use_batchnorm': False,
                        'dropout_rate': 0.0,
                        'test_acc': result.get('test_acc', 0.0),
                        'test_loss': result.get('test_loss', 0.0),
                        'train_time': result.get('train_time', 0.0),
                        'best_val_acc': max(result.get('history', {}).get('val_acc', [0.0]))
                    })
                    print(f"   ✅ Result: Acc={result['test_acc']:.2f}%, Time={result['train_time']:.2f}s")
                else:
                    print(f"   ⚠️ Result missing test_acc, skipping...")
                    failed_experiments += 1
                
            except Exception as e:
                print(f"   ❌ Error in experiment: {e}")
                print(f"   {traceback.format_exc()}")
                failed_experiments += 1
                continue
    
    # ============ Section 2: BatchNorm Analysis ============
    
    print("\n" + "="*70)
    print("🔍 SECTION 2: BATCH NORMALIZATION ANALYSIS")
    print("="*70)
    print("Testing: With BN vs Without BN")
    print("Network depths: Shallow, Medium, Deep")
    print("-"*70)
    
    config.OPTIMIZER = 'adam'  # Reset to default
    
    for use_bn in batchnorm_options:
        for layer in layer_configs:
            model_config = {
                'hidden_sizes': layer['hidden_sizes'],
                'use_batchnorm': use_bn,
                'dropout_rate': 0.0
            }
            
            experiment_counter += 1
            bn_status = "With BN" if use_bn else "Without BN"
            print(f"\n🔬 Experiment {experiment_counter}/34: {bn_status} + {layer['name']}")
            print(f"   Config: hidden_sizes={layer['hidden_sizes']}, BN={use_bn}, Dropout=0.0")
            
            try:
                result = run_experiment(config, model_config)
                
                if result and 'test_acc' in result:
                    all_results.append({
                        'experiment_id': experiment_counter,
                        'category': 'batchnorm',
                        'optimizer': 'adam',
                        'layers': layer['name'],
                        'hidden_sizes': str(layer['hidden_sizes']),
                        'use_batchnorm': use_bn,
                        'dropout_rate': 0.0,
                        'test_acc': result.get('test_acc', 0.0),
                        'test_loss': result.get('test_loss', 0.0),
                        'train_time': result.get('train_time', 0.0),
                        'best_val_acc': max(result.get('history', {}).get('val_acc', [0.0]))
                    })
                    print(f"   ✅ Result: Acc={result['test_acc']:.2f}%, Time={result['train_time']:.2f}s")
                else:
                    print(f"   ⚠️ Result missing test_acc, skipping...")
                    failed_experiments += 1
                
            except Exception as e:
                print(f"   ❌ Error in experiment: {e}")
                print(f"   {traceback.format_exc()}")
                failed_experiments += 1
                continue
    
    # ============ Section 3: Dropout Analysis ============
    
    print("\n" + "="*70)
    print("🔍 SECTION 3: DROPOUT ANALYSIS")
    print("="*70)
    print("Testing: Dropout rates 0.0, 0.3, 0.5")
    print("Network depths: Shallow, Medium, Deep")
    print("-"*70)
    
    config.OPTIMIZER = 'adam'  # Reset to default
    
    for drop_rate in dropout_rates:
        for layer in layer_configs:
            model_config = {
                'hidden_sizes': layer['hidden_sizes'],
                'use_batchnorm': False,
                'dropout_rate': drop_rate
            }
            
            experiment_counter += 1
            print(f"\n🔬 Experiment {experiment_counter}/34: Dropout={drop_rate} + {layer['name']}")
            print(f"   Config: hidden_sizes={layer['hidden_sizes']}, BN=False, Dropout={drop_rate}")
            
            try:
                result = run_experiment(config, model_config)
                
                if result and 'test_acc' in result:
                    all_results.append({
                        'experiment_id': experiment_counter,
                        'category': 'dropout',
                        'optimizer': 'adam',
                        'layers': layer['name'],
                        'hidden_sizes': str(layer['hidden_sizes']),
                        'use_batchnorm': False,
                        'dropout_rate': drop_rate,
                        'test_acc': result.get('test_acc', 0.0),
                        'test_loss': result.get('test_loss', 0.0),
                        'train_time': result.get('train_time', 0.0),
                        'best_val_acc': max(result.get('history', {}).get('val_acc', [0.0]))
                    })
                    print(f"   ✅ Result: Acc={result['test_acc']:.2f}%, Time={result['train_time']:.2f}s")
                else:
                    print(f"   ⚠️ Result missing test_acc, skipping...")
                    failed_experiments += 1
                
            except Exception as e:
                print(f"   ❌ Error in experiment: {e}")
                print(f"   {traceback.format_exc()}")
                failed_experiments += 1
                continue
    
    # ============ Section 4: Hyperparameter Tuning ============
    
    print("\n" + "="*70)
    print("🔍 SECTION 4: HYPERPARAMETER TUNING")
    print("="*70)
    print("Testing: Learning Rate, Batch Size, Weight Decay")
    print("Base model: Medium depth + BN + Dropout(0.3)")
    print("-"*70)
    
    config.OPTIMIZER = 'adam'  # Reset to default
    
    base_model_config = {
        'hidden_sizes': [256, 128],
        'use_batchnorm': True,
        'dropout_rate': 0.3
    }
    
    for i, hp in enumerate(hp_tuning):
        # Update config with new values
        config.LEARNING_RATE = hp.get('lr', 0.001)
        config.BATCH_SIZE = hp.get('batch_size', 128)
        config.WEIGHT_DECAY = hp.get('weight_decay', 1e-4)
        
        experiment_counter += 1
        print(f"\n🔬 Experiment {experiment_counter}/34: HP Config {i+1}")
        print(f"   LR={config.LEARNING_RATE}, BS={config.BATCH_SIZE}, WD={config.WEIGHT_DECAY}")
        
        try:
            result = run_experiment(config, base_model_config)
            
            if result and 'test_acc' in result:
                all_results.append({
                    'experiment_id': experiment_counter,
                    'category': 'hp_tuning',
                    'optimizer': 'adam',
                    'layers': 'Medium',
                    'hidden_sizes': str(base_model_config['hidden_sizes']),
                    'use_batchnorm': base_model_config['use_batchnorm'],
                    'dropout_rate': base_model_config['dropout_rate'],
                    'learning_rate': config.LEARNING_RATE,
                    'batch_size': config.BATCH_SIZE,
                    'weight_decay': config.WEIGHT_DECAY,
                    'test_acc': result.get('test_acc', 0.0),
                    'test_loss': result.get('test_loss', 0.0),
                    'train_time': result.get('train_time', 0.0),
                    'best_val_acc': max(result.get('history', {}).get('val_acc', [0.0]))
                })
                print(f"   ✅ Result: Acc={result['test_acc']:.2f}%, Time={result['train_time']:.2f}s")
            else:
                print(f"   ⚠️ Result missing test_acc, skipping...")
                failed_experiments += 1
            
        except Exception as e:
            print(f"   ❌ Error in experiment: {e}")
            print(f"   {traceback.format_exc()}")
            failed_experiments += 1
            continue
    
    # ============ Save Results ============
    
    total_time = time.time() - start_time_total
    
    # Check if we have any results
    if len(all_results) == 0:
        print("\n❌ No results were collected! Please check for errors above.")
        print("Common issues:")
        print("1. Dataset download failed - try using MNIST")
        print("2. Missing dependencies - run: pip install -r requirements.txt")
        print("3. Insufficient memory - reduce BATCH_SIZE in config.py")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = 'results/experiment_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Results saved to: {csv_path}")
    print(f"📊 Total successful experiments: {len(df)}")
    print(f"📊 Failed experiments: {failed_experiments}")
    
    # Save summary statistics
    summary_path = 'results/summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NEURAL NETWORK PARAMETER ANALYSIS - SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total experiments attempted: {experiment_counter}\n")
        f.write(f"Successful experiments: {len(df)}\n")
        f.write(f"Failed experiments: {failed_experiments}\n")
        f.write(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
        
        if len(df) > 0:
            f.write("BEST PERFORMING CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            best_idx = df['test_acc'].idxmax()
            best_exp = df.loc[best_idx]
            for col in best_exp.index:
                if col not in ['experiment_id', 'category']:
                    f.write(f"  {col}: {best_exp[col]}\n")
            
            f.write("\n\nPERFORMANCE BY CATEGORY:\n")
            f.write("-"*40 + "\n")
            for category in df['category'].unique():
                cat_data = df[df['category']==category]
                f.write(f"\n{category.upper()}:\n")
                f.write(f"  Mean Accuracy: {cat_data['test_acc'].mean():.2f}%\n")
                f.write(f"  Best Accuracy: {cat_data['test_acc'].max():.2f}%\n")
                f.write(f"  Mean Time: {cat_data['train_time'].mean():.2f}s\n")
    
    print(f"📊 Summary saved to: {summary_path}")
    
    # ============ Final Summary ============
    
    print("\n" + "="*70)
    if len(df) > 0:
        print("✅ EXPERIMENTS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️ NO SUCCESSFUL EXPERIMENTS!")
    print("="*70)
    print(f"📊 Total experiments attempted: {experiment_counter}")
    print(f"📊 Successful: {len(df)}")
    print(f"📊 Failed: {failed_experiments}")
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"📁 Results saved in 'results/' directory")
    print("="*70)
    
    if len(df) > 0:
        # Show top 5 results
        print("\n🏆 TOP 5 PERFORMING CONFIGURATIONS:")
        print("-"*70)
        top_5 = df.nlargest(5, 'test_acc')[['experiment_id', 'category', 'test_acc', 'train_time']]
        print(top_5.to_string(index=False))
        
        # Show best config
        print("\n🌟 BEST CONFIGURATION FOUND:")
        print("-"*70)
        best_exp = df.loc[df['test_acc'].idxmax()]
        print(f"  Experiment ID: {best_exp['experiment_id']}")
        print(f"  Category: {best_exp['category']}")
        print(f"  Test Accuracy: {best_exp['test_acc']:.2f}%")
        print(f"  Training Time: {best_exp['train_time']:.2f}s")
        print("\n  Parameters:")
        for col in best_exp.index:
            if col not in ['experiment_id', 'category', 'test_acc', 'test_loss', 'train_time', 'best_val_acc']:
                print(f"    - {col}: {best_exp[col]}")
        
        print("\n" + "="*70)
        print("🎉 PROJECT COMPLETED! Run 'analysis.ipynb' for detailed visualizations.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️ TROUBLESHOOTING TIPS:")
        print("="*70)
        print("1. Check internet connection for dataset download")
        print("2. Try using MNIST: Set DATASET = 'mnist' in config.py")
        print("3. Ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("4. Check if data/ directory has write permissions")
        print("5. Reduce number of experiments by commenting out sections")
        print("="*70)
    
    return df

if __name__ == "__main__":
    try:
        # Run all experiments
        results_df = run_all_experiments()
        
        # Verify results were saved
        if results_df is not None and len(results_df) > 0:
            if os.path.exists('results/experiment_results.csv'):
                print("\n✅ Verification: Results file created successfully!")
                print(f"   File size: {os.path.getsize('results/experiment_results.csv')} bytes")
            else:
                print("\n⚠️ Warning: Results file not found!")
        else:
            print("\n❌ No results to verify. Please check errors above.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user!")
        if os.path.exists('results/experiment_results.csv'):
            print("Partial results have been saved to results/")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print(f"\n{traceback.format_exc()}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check internet connection for dataset download")
        print("2. Try using MNIST: Set DATASET = 'mnist' in config.py")
        print("3. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("4. Check if data/ directory has write permissions") 
