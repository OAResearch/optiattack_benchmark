# OptiAttack Benchmark

This repository contains the benchmark dataset and evaluation framework for the OptiAttack project. It provides a standardized way to evaluate and compare different deep learning models' performance under various attack scenarios.

## Project Structure

The project includes benchmark datasets for several popular deep learning models:
- ResNet50
- ResNet101
- ResNet152
- VGG16
- VGG19
- MobileNetV2
- ShuffleNetV2
- EfficientNet-Lite4

## Requirements

The project requires Python 3.10 or higher and the following main dependencies:
- numpy
- pandas
- onnxruntime
- fastapi
- uvicorn
- pytest

For a complete list of dependencies, please refer to `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OAResearch/optiattack_benchmark.git
cd optiattack_benchmark
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Each model directory contains:
- Pre-trained model weights
- Benchmark datasets
- Evaluation scripts
- Performance metrics

To run benchmarks for a specific model, navigate to the model's directory and follow the instructions in its README file.

## Funding
This work was supported by the Erciyes University Scientific Research Fund (ERU-BAP, Project No: FBA-2024-13536).
## Contributing

Contributions to improve the benchmark dataset or add new models are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request with a detailed description of your changes
