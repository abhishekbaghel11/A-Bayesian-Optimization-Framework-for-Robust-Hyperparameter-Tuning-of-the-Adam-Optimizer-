import torch
import torch.nn as nn
import numpy as np
import gc
import optuna
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.distributions import Normal
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel

from models import CNNNetwork
from train import train_model, extract_features

class AnalyticEI(torch.nn.Module):
    def __init__(self, model, best_f, epsilon=1e-9):
        super().__init__()
        self.model  = model
        self.best_f = best_f
        self._normal = Normal(0, 1)
        self.epsilon = epsilon

    def forward(self, X):
        X_sq = X.squeeze(-2)
        self.model.eval()
        posterior = self.model.posterior(X_sq)
        mu    = posterior.mean.squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)

        improvement = self.best_f - mu
        z           = improvement / sigma

        Phi = self._normal.cdf(z)
        phi = self._normal.log_prob(z).exp()

        ei = improvement * Phi + sigma * phi
        ei = ei.clamp_min(0.0)
        return torch.log(ei + self.epsilon)

def make_analytic_ei_candidates_func(kernel: str, seed: int = 1):
    kernel = kernel.lower()
    def _candidates_func(train_x, train_obj, train_con, bounds, pending_x):
        train_x_n   = normalize(train_x, bounds=bounds)
        if kernel == "matern":
            covar = ScaleKernel(MaternKernel(nu=2.5))
        else:
            covar = ScaleKernel(RBFKernel())

        model = SingleTaskGP(train_x_n, train_obj, covar_module=covar, outcome_transform=Standardize(m=train_obj.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        model.eval()
        best_f_std = model.train_targets.min()
        acqf = AnalyticEI(model=model, best_f=best_f_std)

        standard_bounds = torch.zeros_like(bounds)
        standard_bounds[1] = 1

        with manual_seed(seed):
            candidates, _ = optimize_acqf(
                acq_function=acqf, bounds=standard_bounds, q=1, num_restarts=20,
                raw_samples=1024, options={"batch_limit": 5, "maxiter": 200}, sequential=True,
            )

        return unnormalize(candidates, bounds=bounds)
    return _candidates_func

def objective(trial, train_loader, val_loader, device, l_bounds, u_bounds, wt=0.5, dataset_name="MNIST", train_epochs=1, prune_epochs=None, obj_fn="loss_only"):
    beta1 = trial.suggest_float("beta1", l_bounds[0], u_bounds[0])
    beta2 = trial.suggest_float("beta2", l_bounds[1], u_bounds[1])
    lr = 0.001
    
    model = CNNNetwork(dataset_name=dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-07)
    criterion = nn.CrossEntropyLoss()

    try:
        def cb(epoch, history):
            if prune_epochs and (epoch + 1) == prune_epochs:
                val_loss = history["val_loss"][-1]
                trial.report(val_loss, step=prune_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
        prune_callback = cb if prune_epochs else None
                
        history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=train_epochs, device=device, on_epoch_end=prune_callback)

        val_loss = float(history["val_loss"][-1])
        val_acc  = float(history["val_acc"][-1])
        if np.isnan(val_loss) or np.isinf(val_loss):
            return 1e6

        if obj_fn == "classifier_augmented":
            train_features, train_labels = extract_features(model, train_loader, device)
            val_features, val_labels     = extract_features(model, val_loader, device)

            scaler = StandardScaler().fit(train_features)
            train_features = scaler.transform(train_features)
            val_features   = scaler.transform(val_features)

            if np.isnan(train_features).any() or np.isnan(val_features).any():
                return 1e6
            
            best_clf_acc = 0.0
            for clf in [SVC(max_iter=100), GaussianNB(), LinearDiscriminantAnalysis()]:
                try:
                    c = clf.fit(train_features, train_labels)
                    best_clf_acc = max(best_clf_acc, accuracy_score(val_labels, c.predict(val_features)))
                except: pass

            F = wt * (1.0 - best_clf_acc) + (1.0 - wt) * val_loss
        elif obj_fn == "loss_only":
            F = val_loss

        else:
            raise ValueError(f"The following objective function is not defined.")

        trial.set_user_attr("val_acc", val_acc)
        trial.set_user_attr("val_loss", val_loss)
        return 1e6 if (np.isnan(F) or np.isinf(F)) else float(F)

    finally:
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

def report_best_so_far(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    bt = study.best_trial
    best_acc = bt.user_attrs.get("val_acc", None)
    best_acc_str = f"{best_acc:.4f}" if isinstance(best_acc, (int, float)) else str(best_acc)
    print(f"[After trial {trial.number}] Best so far -> trial={bt.number}, F={bt.value:.6f}, val_acc={best_acc_str}")