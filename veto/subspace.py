"""The text-only subspace and the projection that keeps the LoRA update out of it."""
from __future__ import annotations

import numpy as np
import torch


###### Identifying the shortcut subspace: second moment of the text-only activations #######
class SecondMomentCollector:
    """Accumulates the uncentered second moment of the input of every adapted module.

    Under the text-only input every position is a text position, so all positions count.
    """

    def __init__(self, lora_mods, cov_device="cuda"):
        self.sum = {k: None for k in lora_mods}
        self.n = {k: 0 for k in lora_mods}
        self.cov_device = cov_device
        self.handles = [info["module"].register_forward_hook(self._hook(k))
                        for k, info in lora_mods.items()]

    def _hook(self, key):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x[0].detach().float()
            xx = x.T @ x
            if self.cov_device != x.device.type:
                xx = xx.to(self.cov_device)
            if self.sum[key] is None:
                self.sum[key] = xx
            else:
                self.sum[key].add_(xx)
            self.n[key] += x.shape[0]
        return hook

    def second_moment(self):
        """Divides in place, so call once."""
        return {k: self.sum[k].div_(self.n[k]) for k in self.sum if self.sum[k] is not None}

    def remove(self):
        for h in self.handles:
            h.remove()


############## Identifying the shortcut subspace: leading eigenvectors (rho) ###############
@torch.no_grad()
def leading_subspace(C, rho):
    """Leading eigenvectors of C that carry a fraction `rho` of its spectrum."""
    try:
        ev, V = torch.linalg.eigh(C)
    except RuntimeError:
        ev, V = torch.linalg.eigh(C.double())
        ev, V = ev.float(), V.float()
    ev = ev.flip(0).clamp(min=0)
    V = V.flip(1)
    cum = ev.cumsum(0) / ev.sum()
    r = int((cum < rho).sum().item()) + 1
    return V[:, :r].contiguous()


@torch.no_grad()
def subspaces(moments, rho):
    return {k: leading_subspace(C, rho).cpu() for k, C in moments.items()}


###################### Diagnostics: coverage and overlap of subspaces ######################
@torch.no_grad()
def coverage(Us, moments):
    """Fraction of the activation energy in `moments` that lies inside each subspace."""
    out = {}
    for k, U in Us.items():
        C = moments[k]
        Ud = U.to(C.device)
        out[k] = (torch.trace(Ud.T @ C @ Ud) / torch.trace(C)).item()
    return out


@torch.no_grad()
def overlap(Ua, Ub):
    """||Ua^T Ub||_F^2 / min(rank), per module."""
    out = {}
    for k in Ua:
        a, b = Ua[k].float(), Ub[k].float()
        out[k] = ((a.T @ b) ** 2).sum().item() / min(a.shape[1], b.shape[1])
    return out


########################## Saving and loading the subspace bases ###########################
def save_projectors(path, Us):
    np.savez(path, **{k: v.numpy() for k, v in Us.items()})


def load_projectors(path):
    npz = np.load(path)
    return {k: torch.from_numpy(npz[k]) for k in npz.files}


########### VETO constraint: project the LoRA factor A off the shortcut subspace ###########
@torch.no_grad()
def project_lora(model, projectors, lora_mods):
    """A <- A - (A U) U^T for every adapted module, so that (B A) U = 0."""
    params = dict(model.named_parameters())
    for key, info in lora_mods.items():
        U = projectors.get(key)
        A = params.get(info["param_A_name"])
        if U is None or A is None:
            continue
        U = U.to(A.device, dtype=A.dtype)
        A.data.sub_((A.data @ U) @ U.T)


############### Diagnostic: part of A that lies inside the shortcut subspace ###############
@torch.no_grad()
def subspace_fraction(model, projectors, lora_mods):
    """||A U U^T||_F / ||A||_F averaged over modules. Zero under the constraint."""
    params = dict(model.named_parameters())
    vals = []
    for key, info in lora_mods.items():
        U = projectors.get(key)
        A = params.get(info["param_A_name"])
        if U is None or A is None:
            continue
        A = A.data.float()
        U = U.to(A.device)
        vals.append(((A @ U) @ U.T).norm().item() / max(A.norm().item(), 1e-10))
    return sum(vals) / max(len(vals), 1)
