import torch

from gsinverse.simulator import _LAPLACIAN_KERNEL, create_initial_fields, simulate_gray_scott


def test_laplacian_kernel_sums_to_zero():
    total = sum(sum(row) for row in _LAPLACIAN_KERNEL)
    assert abs(total) < 1e-8


def test_create_initial_fields_shape():
    u, v = create_initial_fields(grid_length=64, patch_radius=2, patch_prob=0.5, seed=0)
    assert u.shape == (1, 1, 64, 64)
    assert v.shape == (1, 1, 64, 64)


def test_create_initial_fields_deterministic_with_seed():
    u1, v1 = create_initial_fields(grid_length=64, patch_radius=2, patch_prob=0.5, seed=123)
    u2, v2 = create_initial_fields(grid_length=64, patch_radius=2, patch_prob=0.5, seed=123)
    assert torch.equal(u1, u2)
    assert torch.equal(v1, v2)


def test_simulate_gray_scott_output_shapes():
    u, v = simulate_gray_scott(
        du=0.16, dv=0.08, f=0.04, k=0.06, iterations=5, size=32, seed=0
    )
    assert u.shape == (32, 32)
    assert v.shape == (32, 32)


def test_simulate_gray_scott_with_initial_fields():
    u, v, u0, v0 = simulate_gray_scott(
        du=0.16, dv=0.08, f=0.04, k=0.06, iterations=5, size=32, seed=0, return_initial=True
    )
    assert u.shape == (32, 32)
    assert v.shape == (32, 32)
    assert u0.shape == (32, 32)
    assert v0.shape == (32, 32)


def test_simulate_gray_scott_deterministic_with_seed():
    u1, v1 = simulate_gray_scott(du=0.16, dv=0.08, f=0.04, k=0.06, iterations=10, size=32, seed=42)
    u2, v2 = simulate_gray_scott(du=0.16, dv=0.08, f=0.04, k=0.06, iterations=10, size=32, seed=42)
    assert torch.equal(u1, u2)
    assert torch.equal(v1, v2)


def test_simulate_gray_scott_no_nans():
    u, v = simulate_gray_scott(du=0.16, dv=0.08, f=0.04, k=0.06, iterations=50, size=32, seed=0)
    assert torch.isfinite(u).all()
    assert torch.isfinite(v).all()
