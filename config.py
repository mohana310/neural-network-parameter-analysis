# config.py
# Central configuration for the project

class Config:
    # Dataset settings  
    DATASET = ' mnist'  # Options: 'mnist','cifar10'
    BATCH_SIZE = 128
    NUM_WORKERS = 2 
    
    # Training settings
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
    
    # Model settings
    INPUT_SIZE = 784  # 28*28 for Fashion-MNIST/MNIST
    NUM_CLASSES = 10
    HIDDEN_SIZES = [128, 256, 128]  # Default for 3-layer model
    
    # Experiment settings
    SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.1
    
    # Paths
    RESULTS_DIR = 'results'
    MODELS_DIR = 'results/models'
    LOGS_DIR = 'results/logs'