# DSD-CRNN

A barebones PyTorch project structure for developing deep learning models. This repository provides a foundation for building and training neural networks, including a CRNN (Convolutional Recurrent Neural Network) architecture.

## Project Structure

```
DSD-CRNN/
├── src/
│   ├── models/           # Model architectures
│   │   ├── __init__.py
│   │   ├── base_model.py # Base model class
│   │   └── crnn.py       # CRNN model (unimplemented)
│   ├── data/             # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py        # Custom dataset class (unimplemented)
│   │   ├── data_loader.py    # Data loader utilities (unimplemented)
│   │   └── synthetic_data.py # Synthetic data generation (outline only)
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── config.py     # Configuration management (unimplemented)
│       └── training.py   # Training utilities (unimplemented)
├── tests/                # Test suite
│   ├── models/
│   ├── data/
│   ├── test_models.py
│   └── test_data.py
├── configs/              # Configuration files
│   └── default_config.yaml
├── train.py              # Training script (skeleton only)
├── evaluate.py           # Evaluation script (skeleton only)
├── requirements.txt      # Project dependencies
└── README.md

```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tiernan7/DSD-CRNN.git
cd DSD-CRNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training (To Be Implemented)

The training script provides a skeleton structure:

```bash
python train.py
```

Expected implementation steps:
1. Load configuration from `configs/default_config.yaml`
2. Initialize model from `src.models`
3. Load data using `src.data`
4. Set up training loop with `src.utils.training`
5. Save checkpoints and logs

### Evaluation (To Be Implemented)

The evaluation script provides a skeleton structure:

```bash
python evaluate.py
```

### Testing

Run the test suite:

```bash
pytest tests/
```

## Synthetic Data Generation (Outline)

The `src/data/synthetic_data.py` module provides an outline for generating synthetic training data. Key features to implement:

- **SyntheticDataGenerator**: Main class for generating synthetic samples
  - `generate_sample()`: Create a single data sample
  - `generate_batch()`: Create a batch of samples
  - `generate_dataset()`: Generate a complete dataset
  - `add_augmentation()`: Apply data augmentation

- **Configuration**: Control synthetic data through `configs/default_config.yaml`:
  ```yaml
  synthetic_data:
    enabled: false
    data_shape: [1, 32, 128]  # [channels, height, width]
    num_classes: 10
    num_train_samples: 1000
    num_val_samples: 200
    num_test_samples: 200
    noise_level: 0.1
  ```

## Models (Unimplemented)

### BaseModel
Base class that all models inherit from, providing:
- Common interface for all models
- Checkpoint saving/loading functionality

### CRNN (Convolutional Recurrent Neural Network)
Placeholder for CRNN architecture combining:
- Convolutional layers for feature extraction
- Recurrent layers (LSTM/GRU) for sequence modeling
- Fully connected layers for classification

## Configuration

Edit `configs/default_config.yaml` to customize:
- Model architecture parameters
- Data loading settings
- Training hyperparameters
- Synthetic data generation parameters

## Development Status

This is a barebones project structure. All core functionality is outlined but not implemented:

- ✅ Project structure created
- ✅ Base classes defined
- ✅ Synthetic data generation outlined
- ⬜ Model implementations (TODO)
- ⬜ Data loading implementation (TODO)
- ⬜ Training loop implementation (TODO)
- ⬜ Evaluation implementation (TODO)

## Contributing

When implementing features:
1. Follow the existing code structure
2. Add corresponding tests in `tests/`
3. Update this README with usage examples
4. Ensure code follows PEP 8 style guidelines

## License

MIT License (or specify your license)

## Contact

For questions or suggestions, please open an issue on GitHub.