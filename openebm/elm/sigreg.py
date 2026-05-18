import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch.distributed.nn.functional as dist_nn_func
except ImportError:
    dist_nn_func = None


def _ddp_sum(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    if dist_nn_func is not None and hasattr(dist_nn_func, "all_reduce"):
        return dist_nn_func.all_reduce(tensor, op=dist.ReduceOp.SUM)

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def compute_sigreg_ddp(hidden, num_slices=1024, num_points=17, eps=1e-6, max_slices_per_chunk=128):
    if hidden.ndim != 2:
        raise ValueError(f"SIGReg expects hidden with shape [N, D], got {tuple(hidden.shape)}")
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    hidden = hidden.float()
    num_samples, dim = hidden.shape
    if num_samples == 0:
        raise ValueError("SIGReg received an empty hidden tensor")

    device = hidden.device
    dtype = hidden.dtype
    points = torch.linspace(-2.0, 2.0, num_points, device=device, dtype=dtype)
    target_cos = torch.exp(-0.5 * points.square())
    global_count = _ddp_sum(torch.tensor(float(num_samples), device=device, dtype=dtype)).clamp_min(eps)

    loss_sum = hidden.new_zeros(())
    slices_done = 0
    while slices_done < num_slices:
        chunk = min(max_slices_per_chunk, num_slices - slices_done)
        directions = torch.randn(chunk, dim, device=device, dtype=dtype)
        directions = F.normalize(directions, p=2, dim=1, eps=eps)

        projections = hidden @ directions.t()
        angles = projections.unsqueeze(-1) * points
        cos_mean = _ddp_sum(torch.cos(angles).sum(dim=0)) / global_count
        sin_mean = _ddp_sum(torch.sin(angles).sum(dim=0)) / global_count

        loss_sum = loss_sum + (cos_mean - target_cos).square().sum() + sin_mean.square().sum()
        slices_done += chunk

    return loss_sum / float(num_slices * num_points)
