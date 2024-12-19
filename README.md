# Stats-mdls-comp-mthds-2024fall
This is the repository for 统计模型与计算方法，2024fall.
## code structure (optional)

algorithms/
    └── conformal.py          # Implementation of weighted split CP (https://arxiv.org/abs/1904.06019)

architectures/
    └── simple_mnistNet.py    # Simple CNN for MNIST classification

baseclasses/
    └── baseclass.py          # Inherit trainers and dataloaders from here

configs/
    └── (tbd)configs.yaml        # Experiment parameters; configuration files

dataloaders/
    └── dataloaders.py        # Loading and preprocessing various datasets

datasets/
    ├── MNIST/               # Directory for MNIST dataset
    └── data-tibshirani.txt   # Other dataset files

trainers/
    └── trainer_simple_mnist.py  # Optimizer loop for simple_mnistNet
