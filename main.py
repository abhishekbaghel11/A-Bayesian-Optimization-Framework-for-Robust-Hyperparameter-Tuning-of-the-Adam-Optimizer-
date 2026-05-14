import argparse
import optuna
try:
    from optuna_integration import BoTorchSampler
except ImportError:
    from optuna.integration import BoTorchSampler
from scipy.stats import qmc

from utils import setup_device
from data import load_dataset
from optimization import objective, report_best_so_far, make_analytic_ei_candidates_func

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization with Optuna and BoTorch")
    parser.add_argument("--subset_method", type=str, default="random_per_class", choices=["random", "random_per_class", "graphcut", "tdds", "none"])
    parser.add_argument("--dataset_name", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "IMBALANCED_CIFAR10", "CIFAR100"])
    parser.add_argument("--kernel", type=str, default="matern", choices=["matern", "rbf"])
    parser.add_argument("--n_startup_trials", type=int, default=25)
    parser.add_argument("--n_trials", type=int, default=125)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--prune_epochs", type=int, default=None)
    parser.add_argument("--wt", type=float, default=0.5)
    parser.add_argument("--obj_fn", type=str, default="loss_only", choices=["loss_only", "classifier_augmented"])
    parser.add_argument("--num_variables", type=int, default=2)
    parser.add_argument("--l_bounds", nargs="+", type=float, default=[0.0, 0.0])
    parser.add_argument("--u_bounds", nargs="+", type=float, default=[0.999, 0.999])
    
    args = parser.parse_args()

    # Hardware Init
    device = setup_device()

    print(f"Loading {args.dataset_name} using Coreset: {args.subset_method}")
    train_loader, val_loader, test_loader, train_subset, val_subset = load_dataset(
        coreset_method=args.subset_method,
        dataset_name=args.dataset_name,
        device=device
    )

    print("\n" + "=" * 80)
    print(f"BoTorchSampler (Analytic logEI) with kernel = {args.kernel.upper()}")
    print("=" * 80)

    sampler = BoTorchSampler(
        candidates_func=make_analytic_ei_candidates_func(kernel=args.kernel, seed=1),
        n_startup_trials=0,
        seed=1,
    )

    pruner = None
    if args.prune_epochs is not None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=args.prune_epochs, interval_steps=1)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    
    # LHS Initialization
    lhs = qmc.LatinHypercube(d=args.num_variables, seed=1)
    lhs_samples = lhs.random(n=args.n_startup_trials)
    scaled_samples = qmc.scale(lhs_samples, args.l_bounds, args.u_bounds)
    
    for row in scaled_samples:
        study.enqueue_trial({"beta1": float(row[0]), "beta2": float(row[1])})
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader, device,
            l_bounds=args.l_bounds, u_bounds=args.u_bounds,
            wt=args.wt, dataset_name=args.dataset_name,
            train_epochs=args.train_epochs, prune_epochs=args.prune_epochs,
            obj_fn=args.obj_fn
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[report_best_so_far],
    )

    # Print Summary
    bt = study.best_trial
    print("\n" + "#" * 80)
    print("Optimization Summary")
    print(
        f"Best Value: F={bt.value:.6f} | Trial={bt.number} | "
        f"beta1={bt.params.get('beta1')} | beta2={bt.params.get('beta2')} | "
        f"val_acc={bt.user_attrs.get('val_acc')}"
    )

if __name__ == "__main__":
    main()