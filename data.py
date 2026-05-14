import os
import sys
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split, DataLoader
from types import SimpleNamespace
from utils import run_cmd, ensure_repo

# Dynamically find the DeepCore folder relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
deepcore_path = os.path.join(current_dir, "DeepCore")

# Append it to sys.path so Python can find the lowercase 'deepcore' module inside it
if deepcore_path not in sys.path:
    sys.path.append(deepcore_path)

try:
    from deepcore.methods import Uniform, Submodular
except ImportError:
    raise ImportError(
        "DeepCore module not found. Please open your terminal and run:\n"
        "git clone https://github.com/PatrickZH/DeepCore.git\n"
        "inside your project directory."
    )

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

def get_random_per_class_indices(train_dataset, fraction, seed):
    rng = np.random.default_rng(seed)
    if hasattr(train_dataset, "targets"):
        targets = np.asarray(train_dataset.targets, dtype=np.int64)
    else:
        base_targets = np.asarray(train_dataset.dataset.targets, dtype=np.int64)
        targets = base_targets[list(train_dataset.indices)]

    classes = np.unique(targets)
    selected = []
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        n_select = max(1, int(np.floor(fraction * len(cls_idx))))
        chosen = rng.choice(cls_idx, size=n_select, replace=False)
        selected.append(chosen)

    all_idx = np.concatenate(selected)
    return all_idx.astype(np.int64)

def make_deepcore_args(dataset_name, device, fraction, seed):
    d_upper = dataset_name.upper()
    if d_upper in ["CIFAR10", "IMBALANCED_CIFAR10"]:
        channel, im_size, num_classes = 3, (32, 32), 10
        class_names = [str(i) for i in range(10)]
    elif d_upper == "CIFAR100":
        channel, im_size, num_classes = 3, (32, 32), 100
        class_names = [str(i) for i in range(100)]
    elif d_upper == "MNIST":
        channel, im_size, num_classes = 1, (28, 28), 10
        class_names = [str(i) for i in range(10)]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    args = SimpleNamespace(
        dataset=dataset_name, model="ResNet18", selection="Submodular", fraction=fraction, seed=seed,
        device=str(device), gpu=None, workers=2, print_freq=50, channel=channel, im_size=im_size,
        num_classes=num_classes, class_names=class_names, batch=256, train_batch=256,
        selection_batch=256, selection_epochs=5, selection_optimizer="SGD", selection_lr=0.1,
        selection_momentum=0.9, selection_weight_decay=5e-4, selection_nesterov=True,
        selection_test_interval=0, selection_test_fraction=1.0, optimizer="SGD", lr=0.1,
        momentum=0.9, weight_decay=5e-4, nesterov=True, min_lr=1e-4, scheduler="CosineAnnealingLR",
        gamma=0.5, step_size=50, balance=True, submodular="GraphCut", submodular_greedy="LazyGreedy",
        uncertainty="Entropy",
    )
    return args

def get_deepcore_subset_indices(train_dataset, method, dataset_name, device, fraction, seed):
    args = make_deepcore_args(dataset_name, device, fraction, seed)
    if method == "random":
        selector = Uniform(dst_train=train_dataset, args=args, fraction=fraction, random_seed=seed, balance=True, replace=False)
    elif method == "graphcut":
        selector = Submodular(dst_train=train_dataset, args=args, fraction=fraction, random_seed=seed, epochs=5, balance=True, function="GraphCut", greedy="LazyGreedy")
    else:
        raise ValueError(f"Unknown deepcore subset method: {method}")
    result = selector.select()
    return np.asarray(result["indices"], dtype=np.int64).reshape(-1)

def generate_tdds_indices(
    keep_ratio=0.10,
    data_dir="./data",
    repo_dir="./external/Dataset-Pruning-TDDS",
    work_dir="./external/Dataset-Pruning-TDDS/checkpoint/mnist_generated",
    epochs=5,
    dataset_name="MNIST"
):
    """Downloads the appropriate TDDS repository and executes its selection scripts."""
    dataset_upper = dataset_name.upper()
    
    if dataset_upper == "MNIST":
        ensure_repo(repo_dir, "https://github.com/Shorya1835/Dataset-Pruning-TDDS.git")
        dataset_used = "mnist"
    elif dataset_upper in ["CIFAR10", "IMBALANCED_CIFAR10"]:
        work_dir = "./external/Dataset-Pruning-TDDS/checkpoint/cifar10_generated"
        ensure_repo(repo_dir, "https://github.com/Shorya1835/Dataset-Pruning-TDDS.git")
        dataset_used = "cifar10"
    elif dataset_upper == "CIFAR100":
        work_dir = "./external/Dataset-Pruning-TDDS/checkpoint/cifar100_generated"
        ensure_repo(repo_dir, "https://github.com/zhangxin-xd/Dataset-Pruning-TDDS.git")
        dataset_used = "cifar100"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for TDDS.")
        
    keep_idx_path = os.path.join(work_dir, f"{dataset_used}_keep_idx_10pct.npy")
    if os.path.exists(keep_idx_path):
        return np.load(keep_idx_path)

    full_ckpt_dir = os.path.join(work_dir, f"{dataset_used}-full")
    mask_dir = os.path.join(work_dir, f"generated_mask_{dataset_used}")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Run TDDS Training Phase
    run_cmd([
        "python", "train.py",
        "--data_path", os.path.abspath(data_dir),
        "--dataset", f"{dataset_used}",
        "--arch", "resnet18",
        "--epochs", str(epochs),
        "--learning_rate", "0.1",
        "--batch-size", "100",
        "--dynamics",
        "--save_path", os.path.abspath(full_ckpt_dir),
    ], cwd=repo_dir)

    # Run TDDS Importance Evaluation
    run_cmd([
        "python", "importance_evaluation.py",
        "--dynamics_path", os.path.join(os.path.abspath(full_ckpt_dir), "npy") + "/",
        "--mask_path", os.path.abspath(mask_dir) + "/",
        "--trajectory_len", str(epochs),
        "--window_size", str(epochs//2),
        "--decay", "0.9",
    ], cwd=repo_dir)

    mask_filename = f'data_mask_win{max(1, epochs//2)}_ep{epochs}.npy'
    mask_file = os.path.join(mask_dir, mask_filename)

    arr = np.load(mask_file)
    arr = np.asarray(arr).astype(int)

    n_total = len(arr)
    n_keep = int(n_total * keep_ratio)
    keep_idx = np.sort(arr[:n_keep])

    os.makedirs(work_dir, exist_ok=True)
    np.save(keep_idx_path, keep_idx)
    return keep_idx

def load_dataset(coreset_method, dataset_name, device, subset_fraction=0.10, subset_seed=42, batch_size=128):
    d_upper = dataset_name.upper()
    if d_upper == "MNIST":
        transform_train = transform_test = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif d_upper in ["CIFAR10", "IMBALANCED_CIFAR10", "CIFAR100"]:
        transform_train = transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        if d_upper == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif d_upper == "IMBALANCED_CIFAR10":
            train_dataset = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=0.05, rand_number=42, train=True, download=True, transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif d_upper == "CIFAR100":
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    SUBSET_INDEX_FILE = f"subset_{coreset_method}_{subset_fraction:.2f}_{subset_seed}.npy"

    if coreset_method in ["random", "graphcut"]:
        train_dataset.targets = np.array(train_dataset.targets)
        if os.path.exists(SUBSET_INDEX_FILE):
            subset_idx = np.load(SUBSET_INDEX_FILE)
        else:
            subset_idx = get_deepcore_subset_indices(train_dataset, coreset_method, dataset_name, device, subset_fraction, subset_seed)
            np.save(SUBSET_INDEX_FILE, subset_idx)
        train_dataset = Subset(train_dataset, subset_idx)
        
    elif coreset_method == "random_per_class":
        train_dataset.targets = np.array(train_dataset.targets)
        if os.path.exists(SUBSET_INDEX_FILE):
            subset_idx = np.load(SUBSET_INDEX_FILE)
        else:
            subset_idx = get_random_per_class_indices(train_dataset, subset_fraction, subset_seed)
            np.save(SUBSET_INDEX_FILE, subset_idx)
        train_dataset = Subset(train_dataset, subset_idx)

    elif coreset_method == "tdds":
        print("Running TDDS trajectory selection...")
        keep_idx = generate_tdds_indices(
            keep_ratio=subset_fraction,
            data_dir="./data",
            dataset_name=dataset_name
        )
        train_dataset = Subset(train_dataset, keep_idx.tolist())

    elif coreset_method == "none":
        print("No coreset method applied. Using full dataset.")
        pass

    else:
        raise ValueError(f"Coreset method {coreset_method} not supported.")

    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    return train_loader, val_loader, test_loader, train_subset, val_subset