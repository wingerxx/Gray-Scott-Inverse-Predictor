"""Forward simulator for the Gray-Scott reaction-diffusion model.

This module ports the prototype's simulation routines verbatim: the math
(discrete Laplacian kernel, update equations, multi-patch seeding) is
preserved exactly so that downstream data generation remains consistent
with the original prototype's behavior.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Isotropic 9-point discrete Laplacian kernel. Sums to zero.
_LAPLACIAN_KERNEL = [
    [0.05, 0.20, 0.05],
    [0.20, -1.00, 0.20],
    [0.05, 0.20, 0.05],
]

# Pre-built float32 kernel tensor (device-agnostic); moved to device inside sim calls.
_KERNEL = torch.tensor(_LAPLACIAN_KERNEL, dtype=torch.float32).view(1, 1, 3, 3)


def create_initial_fields(
    grid_length: int,
    patch_radius: int,
    patch_prob: float,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the initial U and V fields for a Gray-Scott simulation.

    U is initialized to all ones and V to all zeros. A grid of candidate
    patch locations is then visited on a stride of ``2*patch_radius + 1``;
    each location is seeded with a square patch (u=0.50, v=0.25) with
    probability ``patch_prob``. Finally, small symmetry-breaking noise is
    added to both fields.

    Args:
        grid_length: Side length L of the (square) simulation grid.
        patch_radius: Radius of each seeded patch.
        patch_prob: Probability of seeding a patch at each candidate site.
        seed: Optional RNG seed. If ``None``, the generator is seeded from
            system entropy.
        device: Torch device on which to create the fields and generator.

    Returns:
        A tuple ``(u, v)`` of tensors with shape ``(1, 1, grid_length, grid_length)``.
    """
    u = torch.ones(1, 1, grid_length, grid_length, device=device)
    v = torch.zeros(1, 1, grid_length, grid_length, device=device)

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.seed()

    stride = patch_radius * 2 + 1
    for x in range(patch_radius, grid_length - patch_radius, stride):
        for y in range(patch_radius, grid_length - patch_radius, stride):
            if torch.rand(1, generator=rng, device=device).item() < patch_prob:
                x0, x1 = x - patch_radius, x + patch_radius + 1
                y0, y1 = y - patch_radius, y + patch_radius + 1
                u[0, 0, x0:x1, y0:y1] = 0.50
                v[0, 0, x0:x1, y0:y1] = 0.25

    u += torch.rand(u.shape, generator=rng, device=device) * 0.05
    v += torch.rand(v.shape, generator=rng, device=device) * 0.05

    return u, v


def simulate_gray_scott(
    du: float,
    dv: float,
    f: float,
    k: float,
    iterations: int = 1000,
    size: int = 64,
    patch_radius: int = 2,
    patch_prob: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cpu",
    return_initial: bool = False,
    u0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
):
    """Run a Gray-Scott reaction-diffusion simulation.

    Args:
        du: Diffusion rate for U.
        dv: Diffusion rate for V.
        f: Feed rate.
        k: Kill rate.
        iterations: Number of simulation steps to run.
        size: Side length of the (square) simulation grid.
        patch_radius: Radius of seeded initial patches.
        patch_prob: Probability of seeding a patch at each candidate site.
        seed: Optional RNG seed for the initial conditions.
        device: Torch device on which to run the simulation.
        return_initial: If ``True``, also return the initial U and V fields
            (as 2D arrays) alongside the final fields.

    Returns:
        If ``return_initial`` is ``False``: a tuple ``(u, v)`` of the final
        ``(size, size)`` fields.
        If ``return_initial`` is ``True``: a tuple
        ``(u, v, u_initial, v_initial)`` where the initial fields are also
        ``(size, size)`` arrays.
    """
    if u0 is not None and v0 is not None:
        u = u0.to(device)
        v = v0.to(device)
    else:
        u, v = create_initial_fields(size, patch_radius, patch_prob, seed=seed, device=device)

    if return_initial:
        u_initial = u.clone().squeeze()
        v_initial = v.clone().squeeze()

    kernel = _KERNEL.to(device=device, dtype=u.dtype)

    dt = 1.0
    for _ in range(iterations):
        u_pad = F.pad(u, (1, 1, 1, 1), mode="circular")
        v_pad = F.pad(v, (1, 1, 1, 1), mode="circular")

        lap_u = F.conv2d(u_pad, kernel)
        lap_v = F.conv2d(v_pad, kernel)

        uvv = u * v * v

        u = u + dt * (du * lap_u - uvv + f * (1 - u))
        v = v + dt * (dv * lap_v + uvv - (f + k) * v)

    u_final = u.squeeze()
    v_final = v.squeeze()

    if return_initial:
        return u_final, v_final, u_initial, v_initial
    return u_final, v_final


def simulate_gray_scott_two_phase(
    du: float,
    dv: float,
    f: float,
    k: float,
    iterations: int = 2000,
    stability_steps: int = 200,
    size: int = 64,
    patch_radius: int = 2,
    patch_prob: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a two-phase simulation to support convergence checking.

    Runs for ``iterations`` steps, captures V at that point (the
    ``snapshot``), then runs ``stability_steps`` more. Returns the
    final V field, the snapshot V field, and the final V field again so
    callers can compare snapshot vs final to assess stability.

    Args:
        du: Diffusion rate for U.
        dv: Diffusion rate for V.
        f: Feed rate.
        k: Kill rate.
        iterations: Number of steps before the stability snapshot.
        stability_steps: Additional steps run after the snapshot.
        size: Side length of the simulation grid.
        patch_radius: Radius of seeded initial patches.
        patch_prob: Probability of seeding a patch at each candidate site.
        seed: Optional RNG seed for the initial conditions.
        device: Torch device on which to run the simulation.

    Returns:
        A tuple ``(u_snapshot, v_snapshot, v_final, v_initial)`` of
        ``(size, size)`` tensors, where the snapshot fields are captured
        after ``iterations`` steps and ``v_final`` is after
        ``iterations + stability_steps`` steps. ``v_initial`` is the V
        field at t=0, before any stepping occurs.
    """
    u, v = create_initial_fields(size, patch_radius, patch_prob, seed=seed, device=device)

    v_initial = v.squeeze().clone()

    kernel = _KERNEL.to(device=device, dtype=u.dtype)

    dt = 1.0

    def _step(u, v):
        u_pad = F.pad(u, (1, 1, 1, 1), mode="circular")
        v_pad = F.pad(v, (1, 1, 1, 1), mode="circular")
        lap_u = F.conv2d(u_pad, kernel)
        lap_v = F.conv2d(v_pad, kernel)
        uvv = u * v * v
        u = u + dt * (du * lap_u - uvv + f * (1 - u))
        v = v + dt * (dv * lap_v + uvv - (f + k) * v)
        return u, v

    for _ in range(iterations):
        u, v = _step(u, v)

    u_snapshot = u.squeeze().clone()
    v_snapshot = v.squeeze().clone()

    for _ in range(stability_steps):
        u, v = _step(u, v)

    v_final = v.squeeze()
    return u_snapshot, v_snapshot, v_final, v_initial
