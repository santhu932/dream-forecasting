import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from metric_utils import bb_confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score


# losses for binary labels
def binary_cross_entropy(y_pred, y_true):
    # can also be applied to grid labels
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='sum') / y_true.size(0)


# losses for grid labels
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2):
    # compute the *weighted* cross entropy loss, see https://arxiv.org/abs/1708.02002.
    # size of y_true is [B, 1, H, W]
    # size of y_pred is [B, 1, H, W]
    return sigmoid_focal_loss(y_pred, y_true, alpha=alpha, gamma=gamma, reduction='sum') / y_true.size(0)


def dice_loss(y_pred, y_true):
    # size of y_true is [B, 1, H, W]
    # size of y_pred is [B, 1, H, W]
    assert y_pred.size() == y_true.size()
    y_pred = torch.sigmoid(y_pred)
    numerator = 2 * torch.sum(y_pred * y_true)
    denominator = torch.sum(y_pred + y_true)
    return 1 - numerator / denominator


def bbiou(y_pred, y_true, **kwargs):
    # Bounding box IoU metric
    # Both y_pred and y_true are numpy arrays.
    N = y_true.shape[0]
    tp, fp, fn = 0, 0, 0
    for i in range(N):
        _tp, _, _fp, _fn = bb_confusion_matrix(y_true[i], y_pred[i], **kwargs)
        tp += _tp
        fp += _fp
        fn += _fn
    return {'TP': tp, 'FP': fp, 'FN': fn}

#losses for VAE:

def compute_rbf(x1, x2, eps = 1e-7, z_var = 2.):
    
    z_dim = x2.size(-1)
    sigma = 2. * z_dim * z_var

    result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    
    return result

def compute_inv_mult_quad(x1, x2, eps = 1e-7, z_var = 2.):
    
    z_dim = x2.size(-1)
    C = 2 * z_dim * z_var
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()

    return result

def compute_kernel(x1, x2, kernel_type = 'rbf'):

    x1 = x1.unsqueeze(-2) # Make it into a column tensor
    x2 = x2.unsqueeze(-3) # Make it into a row tensor

    if kernel_type == 'rbf':
        result = compute_rbf(x1, x2)
    elif kernel_type == 'imq':
        result = compute_inv_mult_quad(x1, x2)
    else:
        raise ValueError('Undefined kernel type.')

    return result

    
def compute_mmd(z):
    # Sample from prior (Gaussian) distribution
    prior_z = torch.randn_like(z).to(device = z.device)

    prior_z__kernel = compute_kernel(prior_z, prior_z)
    z__kernel = compute_kernel(z, z)
    priorz_z__kernel = compute_kernel(prior_z, z)

    mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
    return mmd


def compute_rbf_batched(x1, x2, batch_size=256, z_var=2.):
    """
    Compute RBF kernel in batches to manage memory usage.
    """
    sigma = 2. * x2.size(-1) * z_var
    results = []
    for i in range(0, x1.size(0), batch_size):
        for j in range(0, x2.size(0), batch_size):
            diff = x1[i:i+batch_size] - x2[j:j+batch_size]
            exp_part = torch.exp(-torch.norm(diff, dim=-1, p=2).pow(2) / sigma)
            results.append(exp_part)
    return torch.cat(results, dim=0)

def compute_inv_mult_quad_batched(x1, x2, batch_size=256, eps=1e-7, z_var=2.):
    """
    Compute Inverse Multiquadratic kernel in batches.
    """
    C = 2 * x2.size(-1) * z_var
    results = []
    for i in range(0, x1.size(0), batch_size):
        for j in range(0, x2.size(0), batch_size):
            diff = x1[i:i+batch_size] - x2[j:j+batch_size]
            kernel = C / (eps + C + torch.norm(diff, dim=-1, p=2).pow(2))
            if i == j:
                kernel -= torch.diag_embed(torch.diag(kernel))
            results.append(kernel)
    return torch.cat(results, dim=0)

def compute_kernel_batched(x1, x2, kernel_type='rbf', batch_size=256):
    x1 = x1.unsqueeze(-2) # Make it into a column tensor
    x2 = x2.unsqueeze(-3) # Make it into a row tensor
    
    if kernel_type == 'rbf':
        return compute_rbf_batched(x1, x2, batch_size)
    elif kernel_type == 'imq':
        return compute_inv_mult_quad_batched(x1, x2, batch_size)
    else:
        raise ValueError('Undefined kernel type.')

def compute_mmd_batched(z, batch_size=2):
    prior_z = torch.randn_like(z).to(device=z.device)
    prior_z__kernel = compute_kernel_batched(prior_z, prior_z, batch_size=batch_size)
    z__kernel = compute_kernel_batched(z, z, batch_size=batch_size)
    priorz_z__kernel = compute_kernel_batched(prior_z, z, batch_size=batch_size)

    mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
    return mmd

def balanced_vae_loss(y_pred, y_true, mean, logvar, z, alpha = - 9.0, beta = 10.5, kl_weight = 0.0001, reg_weight = 110):
    z_flattened = z.view(z.size(0), -1)
    batch_size = y_true.size(0) 
    bias_corr = batch_size *  (batch_size - 1)
    recon_loss = mse_recon(y_pred=y_pred, y_true=y_true)
    mmd_loss = compute_mmd_batched(z_flattened)
    kl_loss = - 0.5 * torch.sum(1+ logvar - (mean ** 2) - logvar.exp()) / y_true.size(0) / y_true.size(-1)
    total_loss = beta * recon_loss +  (1-alpha) * kl_weight * kl_loss + (alpha + reg_weight - 1.)/bias_corr * mmd_loss
    #total_loss = beta * recon_loss +  (1-alpha) * kl_weight * kl_loss
    return total_loss

def vae_loss(y_pred, y_true, mean, logvar, z):
    recon_loss = mae_recon(y_pred=y_pred, y_true=y_true)
    #recon_loss = torch.sum((y_pred - y_true) ** 2)
    kl_loss = - 0.5 * torch.sum(1+ logvar - (mean ** 2) - logvar.exp()) / y_true.size(0) / y_true.size(-1)
    beta = 0.0001
    total_loss = recon_loss + beta * kl_loss
    return total_loss

def hierarchical_vae_loss(y_pred, y_true, means, logvars):
    recon_loss = torch.sum(torch.abs(y_true - y_pred)) / y_true.size(0) / y_true.size(-1)
    kl_loss = 0.
    beta = 0.0001
    for i in range(len(means)):
        if i == 1:
            beta = 0.00001
        if i > 1:
            beta = 0.000001
        kl_loss += beta * (- 0.5 * torch.sum(1+ logvars[i] - (means[i] ** 2) - logvars[i].exp()) / y_true.size(0) / y_true.size(-1))
    
    total_loss = recon_loss + kl_loss
    return total_loss

        
# losses for reconstruction
def mae_recon(y_pred, y_true):
    assert y_pred.size() == y_true.size()
    return torch.sum(torch.abs(y_true - y_pred)) / y_true.size(0) / y_true.size(-1) # average over batch and channels
    
def mse_recon(y_pred, y_true):
    assert y_pred.size() == y_true.size()
    return torch.sum((y_pred - y_true) ** 2) / y_true.size(0) / y_true.size(-1) # average over batch and channels

# losses for reconstruction
def log_recon(y_pred, y_true, y_var):
    assert y_pred.size() == y_true.size()
    return 0.5 * torch.sum((y_pred - y_true) ** 2 / y_var + torch.log(y_var)) / y_true.size(0) / y_true.size(-1) # average over batch and channels


def binary_confusion_matrix(y_pred, y_true, threshold=0.5):
    # Convert probabilities to binary predictions based on the threshold
    predictions = (y_pred > threshold).astype(int)
    auroc = roc_auc_score(y_true, y_pred)

    # Calculate confusion matrix
    TP = np.sum((predictions == 1) & (y_true == 1))
    FP = np.sum((predictions == 1) & (y_true == 0))
    FN = np.sum((predictions == 0) & (y_true == 1))
    f1 = 2 * TP / (2 * TP + FP + FN)

    return {'TP': TP, 'FP': FP, 'FN': FN, 'auroc': auroc, 'f1': f1}
