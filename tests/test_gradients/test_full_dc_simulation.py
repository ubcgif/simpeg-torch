"""
Test complete DC simulation workflow with gradient computation.
"""

import torch
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity.simulation import (
    DCStaticSimulationCellCentered,
)

torch.set_default_dtype(torch.float64)


def test_dc_simulation_fields_with_gradients():
    """Test complete DC simulation with field computation and gradients."""

    # Create a simple 2D mesh
    mesh = TensorMesh([4, 4], dtype=torch.float64)

    # Create DC simulation
    sim = DCStaticSimulationCellCentered(mesh)
    sim.setBC()

    # Create resistivity with gradients
    resistivity = torch.full(
        (mesh.n_cells,), 100.0, dtype=torch.float64, requires_grad=True
    )

    # Create a simple source configuration
    class SimpleSource:
        def __init__(self, cell, current):
            self.cell = cell
            self.current = current

    # Add sources (current injection and extraction)
    source_pos = [
        SimpleSource(0, 1.0),
        SimpleSource(-1, -1.0),
    ]  # Inject at first cell, extract at last
    sim.sources = source_pos

    # Compute fields - this tests the complete gradient flow through:
    # 1. Face inner product with inversion
    # 2. System matrix assembly (D @ MfRhoI @ G)
    # 3. Linear system solve with TorchMatSolver
    fields = sim.fields(resistivity)

    # Compute a simple objective function
    loss = torch.sum(fields**2)

    # Compute gradients
    loss.backward()

    # Verify results
    assert fields is not None, "Fields should be computed"
    assert fields.shape[0] == mesh.n_cells, "Fields should have correct shape"
    assert torch.all(torch.isfinite(fields)), "Fields should be finite"

    assert resistivity.grad is not None, "Gradients should be computed"
    assert torch.all(torch.isfinite(resistivity.grad)), "Gradients should be finite"
    assert torch.any(resistivity.grad != 0), "Some gradients should be non-zero"

    print("✅ Complete DC simulation test passed")
    print(f"Mesh cells: {mesh.n_cells}")
    print(f"Fields shape: {fields.shape}")
    print(f"Fields range: [{fields.min():.6f}, {fields.max():.6f}]")
    print(f"Gradient mean: {resistivity.grad.mean():.2e}")
    print(f"Gradient std: {resistivity.grad.std():.2e}")


if __name__ == "__main__":
    test_dc_simulation_fields_with_gradients()
    print("\n🎉 Full DC simulation with gradients works!")
