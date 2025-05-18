import torch
import numpy as np
from .tensor_mesh import TensorMesh


def setup_mesh(mesh_type, nC, nDim=3, random_seed=None):
    # Set up random number generator if needed
    if "random" in mesh_type:
        rng = torch.Generator()
        if isinstance(random_seed, int):
            rng.manual_seed(random_seed)

    if "TensorMesh" in mesh_type:
        if "uniform" in mesh_type:
            h = [torch.full((nC,), 1.0 / nC) for _ in range(nDim)]
        elif "random" in mesh_type:
            h = []
            for _ in range(nDim):
                hi = torch.rand(nC, generator=rng) * nC * 0.5 + nC * 0.5
                hi = hi / hi.sum()  # normalize
                h.append(hi)
        else:
            raise Exception("Unexpected mesh_type")

        # `mesh` placeholder — replace with actual TensorMesh constructor from your code
        mesh = TensorMesh(
            h
        )  # This assumes you have a PyTorch-compatible `TensorMesh` class

        max_h = max(hi.max().item() for hi in mesh.h)
        return mesh, max_h

    raise Exception("Unsupported mesh_type")


class OrderTest:
    """Torch-compatible base class for convergence testing of discrete operators."""

    name = "Order Test"
    expectedOrders = 2.0
    tolerance = 0.85
    meshSizes = [4, 8, 16, 32]
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 3
    random_seed = None

    def setupMesh(self, nC):
        """Generate mesh and set as current mesh for testing.

        Parameters
        ----------
        nC : int
            Number of cells along each axis.

        Returns
        -------
        Float
            Maximum cell width for the mesh
        """
        mesh, max_h = setup_mesh(
            self._meshType, nC, self.meshDimension, random_seed=self.random_seed
        )
        self.M = mesh
        return max_h

    def getError(self):
        return 1.0

    def orderTest(self, random_seed=None):
        if not isinstance(self.meshTypes, list):
            raise TypeError("meshTypes must be a list")
        if isinstance(self.tolerance, (int, float)):
            self.tolerance = [self.tolerance] * len(self.meshTypes)

        if isinstance(self.expectedOrders, (float, int)):
            self.expectedOrders = [self.expectedOrders] * len(self.meshTypes)

        if len(self.expectedOrders) != len(self.meshTypes):
            raise ValueError("expectedOrders must have the same length as meshTypes")

        if random_seed is not None:
            self.random_seed = random_seed

        def test_func(n_cells):
            max_h = self.setupMesh(n_cells)
            err = self.getError()
            return err, max_h

        for mesh_type, order, tol in zip(
            self.meshTypes, self.expectedOrders, self.tolerance
        ):
            self._meshType = mesh_type
            assert_expected_order(
                test_func,
                self.meshSizes,
                expected_order=order,
                rtol=np.abs(1 - tol),
                test_type="mean_at_least",
            )


def assert_expected_order(
    func, n_cells, expected_order=2.0, rtol=0.15, test_type="mean"
):
    """Perform an order test using PyTorch-compatible structures."""
    n_cells = torch.as_tensor(n_cells, dtype=torch.int)
    if test_type not in ["mean", "min", "last", "all", "mean_at_least"]:
        raise ValueError("Unsupported test_type: " + test_type)

    orders = []
    nc = n_cells[0].item()
    err_last, h_last = func(nc)

    print("_______________________________________________________")
    print("  nc  |    h    |    error    | e(i-1)/e(i) |  order   ")
    print("~~~~~~|~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~")
    print(f"{nc:^6d}|{h_last:^9.2e}|{err_last:^13.3e}|             |")

    for nc in n_cells[1:]:
        nc = nc.item()
        err, h = func(nc)
        order = torch.log(torch.tensor(err / err_last)) / torch.log(
            torch.tensor(h / h_last)
        )
        ratio = err_last / err
        print(f"{nc:^6d}|{h:^9.2e}|{err:^13.3e}|{ratio:^13.4f}|{order.item():^10.4f}")
        err_last = err
        h_last = h
        orders.append(order.item())

    print("-------------------------------------------------------")

    orders_tensor = torch.tensor(orders)

    if test_type == "mean":
        np.testing.assert_allclose(
            torch.mean(orders_tensor).item(), expected_order, rtol=rtol
        )
    elif test_type == "mean_at_least":
        if not (torch.mean(orders_tensor) > expected_order * (1 - rtol)):
            raise AssertionError(
                f"\nOrder mean {torch.mean(orders_tensor).item()} is not greater than the expected order "
                f"{expected_order} within the tolerance {rtol}."
            )
    elif test_type == "min":
        np.testing.assert_allclose(
            torch.min(orders_tensor).item(), expected_order, rtol=rtol
        )
    elif test_type == "last":
        np.testing.assert_allclose(orders_tensor[-1].item(), expected_order, rtol=rtol)
    elif test_type == "all":
        np.testing.assert_allclose(orders_tensor.numpy(), expected_order, rtol=rtol)

    return orders_tensor
