import os
import sys
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.append("..")
from lib.datasets import get_dataset
from lib.models import get_model
from lib.utils import get_quick_loader, predict_train2, predict_test, train_network
from lib.variances import get_pred_vars_laplace
from torch.nn.utils import parameters_to_vector


def build_model(model_name, nc, ds_train, device, seed, verbose=True):
    data_arr = getattr(ds_train, 'data', None)
    if data_arr is None:
        raise RuntimeError("Dataset has no attribute .data to infer input shape")
    arr = np.array(data_arr)
    # arr may be (N,H,W,C) or (N,C,H,W)
    if arr.ndim == 4 and arr.shape[-1] in (1,3):
        H, W, C = arr.shape[1], arr.shape[2], arr.shape[3]
    elif arr.ndim == 4 and arr.shape[1] in (1,3):
        C, H, W = arr.shape[1], arr.shape[2], arr.shape[3]
    else:
        x0, _ = ds_train[0]
        if isinstance(x0, torch.Tensor):
            C, H, W = x0.shape
        else:
            raise RuntimeError("Cannot infer input shape from dataset")
    input_size_try = C * H * W

    model = None
    err_msgs = []
    try:
        model = get_model(model_name, nc, input_size_try, device, seed)
        if verbose: print(f"Built model with signature get_model(name, nc, input_size, device, seed)")
    except Exception as e1:
        err_msgs.append(str(e1))
    if model is None:
        try:
            model = get_model(model_name, nc, input_shape=(C,H,W), device=device, seed=seed)
            if verbose: print(f"Built model with signature get_model(name, nc, input_shape=..., device=..., seed=...)")
        except Exception as e2:
            err_msgs.append(str(e2))
    if model is None:
        try:
            model = get_model(model_name, nc)
            if verbose: print("Built model with signature get_model(name, nc)")
        except Exception as e3:
            err_msgs.append(str(e3))
    if model is None:
        try:
            model = get_model(model_name, nc, device=device)
            if verbose: print("Built model with signature get_model(name, nc, device=device)")
        except Exception as e4:
            err_msgs.append(str(e4))

    if model is None:
        raise RuntimeError("Failed to build model. Tried signatures; errors:\n" + "\n".join(err_msgs))

    # ensure model on device
    model = model.to(device)
    # seed model weights if possible (some get_model already sets seed)
    try:
        torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name_exp', default='exp1_cifar10_labelflip_mpe', type=str)
    p.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10'])
    p.add_argument('--model', default='cnn_deepobs')
    p.add_argument('--bs', default=256, type=int)
    p.add_argument('--bs_jacs', default=20, type=int)
    p.add_argument('--epochs', default=250, type=int)
    # p.add_argument('--epochs_retrain', default=150, type=int)
    p.add_argument('--lr', default=0.01, type=float)
    p.add_argument('--lrmin', default=1e-4, type=float)
    p.add_argument('--delta', default=250, type=float)
    p.add_argument('--flip_frac', default=0.05, type=float)
    p.add_argument('--n_repeats', default=5, type=int)
    p.add_argument('--seed0', default=1, type=int)
    return p.parse_args()

def flip_labels_random(targets, indices_to_flip, nc, rng):
    new = np.array(targets).copy()
    for idx in indices_to_flip:
        orig = int(new[idx])
        # choose random different class
        choices = list(range(nc))
        choices.remove(orig)
        new[idx] = rng.choice(choices)
    return new

if __name__ == "__main__":
    args = parse_args()
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    results = []
    os.makedirs('pickles', exist_ok=True)

    for run in range(args.n_repeats):
        seed = args.seed0 + run
        np.random.seed(seed)
        torch.manual_seed(seed)

        # load dataset
        ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True)
        n_train = len(ds_train)
        nc = int(max(ds_train.targets)) + 1
        n_flip = int(round(n_train * args.flip_frac))
        rng = np.random.default_rng(seed)

        # choose indices to flip (ground truth bad labels)
        flip_indices = rng.choice(n_train, size=int(n_train * 0.05), replace=False)

        # create a copy of dataset targets and flip them
        targets_orig = np.array(ds_train.targets).copy()
        targets_flipped = flip_labels_random(targets_orig, flip_indices, nc, rng)

        # assign flipped targets into dataset copy
        ds_train.targets = torch.asarray(targets_flipped)

        # create loaders
        trainloader = get_quick_loader(DataLoader(ds_train, batch_size=args.bs, shuffle=True))
        trainloader_eval = DataLoader(ds_train, batch_size=args.bs, shuffle=False)
        testloader_eval = DataLoader(ds_test, batch_size=args.bs, shuffle=False)
        trainloader_vars = DataLoader(ds_train, batch_size=args.bs_jacs, shuffle=False)

        # build model
        net = build_model(args.model, nc, ds_train, device, seed)
        # train
        net, losses = train_network(net, trainloader, args.lr, args.lrmin, args.epochs, n_train, args.delta)

        # Evaluate training predictions and get residuals/lambdas
        residuals, probs, lambdas, train_acc, train_nll = predict_train2(net, trainloader_eval, nc,
                                                                         torch.asarray(ds_train.targets), device)

        # Compute variances (K-FAC laplace as in original script)
        vars_list = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, version='kfac', device=device)

        # compute sensitivities and ranking
        sensitivities = np.asarray(residuals) * np.asarray(lambdas) * np.asarray(vars_list)
        sensitivities = np.sum(np.abs(sensitivities), axis=-1)

        # predict top n_flip as suspected errors
        detected_idx = np.argsort(-sensitivities)[:n_flip]

        # compute confusion matrix (binary)
        y_true = np.zeros(n_train, dtype=int)
        y_true[flip_indices] = 1
        y_pred = np.zeros(n_train, dtype=int)
        y_pred[detected_idx] = 1

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        print(f"run {run}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        results.append({'run': run, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                        'n_flip': int(n_flip)})

        # restore original labels to avoid side effects for next run
        ds_train.targets = torch.asarray(targets_orig)

    # aggregate
    tps = np.array([r['tp'] for r in results])
    fps = np.array([r['fp'] for r in results])
    fns = np.array([r['fn'] for r in results])
    tns = np.array([r['tn'] for r in results])

    summary = {
        'mean': {'tp': tps.mean(), 'fp': fps.mean(), 'fn': fns.mean(), 'tn': tns.mean()},
        'std': {'tp': tps.std(ddof=1), 'fp': fps.std(ddof=1), 'fn': fns.std(ddof=1), 'tn': tns.std(ddof=1)},
        'per_run': results
    }

    with open(f'pickles/{args.name_exp}_confusion.pkl', 'wb') as f:
        pickle.dump(summary, f)

    print("Summary (mean ± std):")
    for k in ['tp','fp','fn','tn']:
        print(f"{k}: {summary['mean'][k]:.2f} ± {summary['std'][k]:.2f}")
    print("Saved to pickles/")
