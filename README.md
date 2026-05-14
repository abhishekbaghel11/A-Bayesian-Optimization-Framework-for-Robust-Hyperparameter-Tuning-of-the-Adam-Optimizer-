# A Bayesian Optimization Framework for Robust Hyperparameter Tuning of the Adam Optimizer 

This repository contains a modular Python pipeline for optimizing hyperparameters (like Adam's `beta1` and `beta2`) of deep learning models using **Bayesian Optimization**. To accelerate the optimization process, this project integrates data subsetting (coreset selection) techniques.

---

## 📁 Code Files Structure

```text
bayesian_opt_project/
├── requirements.txt   # Python dependencies
├── utils.py           # Device setup and shell command utilities
├── models.py          # CNN model architectures
├── data.py            # Dataset loading, imbalance logic, and Coreset selection
├── train.py           # PyTorch training loops and feature extraction
├── optimization.py    # BoTorch Acquisition functions, Optuna objectives, and callbacks
└── main.py            # CLI entry point for running the optimization study

```

---

## ⚙️ Installation & Setup

### 1. Clone this repository

First, clone this project to your local machine and navigate into the directory:

```bash
git clone https://github.com/abhishekbaghel11/A-Bayesian-Optimization-Framework-for-Robust-Hyperparameter-Tuning-of-the-Adam-Optimizer-.git
cd bayesian_opt_project
```

### 2. Install Python Dependencies

It is recommended to use a virtual environment like conda. Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Clone the DeepCore Dependency

This project relies on the external DeepCore library for data pruning techniques like GraphCut. You must clone it directly into the project root:

```bash
git clone https://github.com/PatrickZH/DeepCore.git
```

---

## 🚀 How to Run

The optimization pipeline is executed entirely through the `main.py` script. You can configure the study using command-line arguments.

### Basic Usage

Example: To run the optimization on CIFAR10 dataset using a "random per-class" coreset method and the Matern kernel:

```bash
python main.py --dataset_name CIFAR10 --subset_method random_per_class --kernel matern
```

### Advanced Usage (Custom Iterations & Bounds)

Example: To run a larger study on Imbalanced CIFAR10 using GraphCut with specific search bounds:

```bash
python main.py \
    --dataset_name IMBALANCED_CIFAR10 \
    --subset_method graphcut \
    --kernel rbf \
    --n_startup_trials 25 \
    --n_trials 100 \
    --train_epochs 5 \
    --l_bounds 0.5 0.5 \
    --u_bounds 0.99 0.99
```

---

## 🎛️ Command-Line Arguments

The following is the list of the arguments that are available for the script:

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `CIFAR10` | Datasets available to use: `MNIST`, `CIFAR10`, `IMBALANCED_CIFAR10`, `CIFAR100` |
| `--subset_method` | `random_per_class` | Coreset methods available: `random`, `random_per_class`, `graphcut`, `tdds`, `none`|
| `--kernel` | `matern` | Gaussian Process kernel : `matern` or `rbf` |
| `--n_startup_trials` | `25` | Number of initial Latin Hypercube Sampling (LHS) trials |
| `--n_trials` | `125` | Total number of optimization trials (including startup trials) |
| `--train_epochs` | `1` | Number of epochs to train the model per trial |
| `--prune_epochs` | `None` | Epoch at which Optuna should check for trial pruning |
| `--obj_fn` | `loss_only` | Objective to minimize: `loss_only` or `classifier_augmented` |
| `--wt` | `0.5` | Weight used if `obj_fn` is `classifier_augmented` |
| `--l_bounds` | `0.0 0.0` | Lower bounds for `beta1` and `beta2` |
| `--u_bounds` | `0.999 0.999` | Upper bounds for `beta1` and `beta2` |
| `--num_variables` | `2` | The number of variables that are being optimized (will need to make changes in the objective function if different value given)

>Note : `none` in the subset_method refers to full dataset being used instead of applying any coreset methods.

---

## 🧠 Notes on TDDS Selection

If you run the pipeline with `--subset_method tdds`, the script will dynamically clone the required TDDS repositories into an `./external/` directory and execute the trajectory generation sub-scripts.

Ensure you have a stable internet connection for the initial run.

---

## 🔧 Custom Objective Functions

This project currently supports two objective functions:

- `loss_only`
- `classifier_augmented`

If you want to experiment with your own optimization criteria, you can modify the `objective()` optimization logic inside `optimization.py` and introduce a custom objective function of your choice. After adding it, you can integrate it into the command-line pipeline and use it during Bayesian Optimization runs.
