import torch
import numpy as np
import pytest

from simpegtorch.discretize.utils import (
    sdiag,
    sub2ind,
    speye,
    ndgrid,
    mkvc,
    kron,
    reshape_fortran,
    is_scalar,
    inverse_2x2_block_diagonal,
    inverse_3x3_block_diagonal,
    inverse_property_tensor,
    make_property_tensor,
    # index_cube,
    ind2sub,
    as_array_n_by_dim,
    TensorType,
    Zero,
    Identity,
    # extract_core_mesh,
    # active_from_xyz,
    # mesh_builder_xyz,
    # refine_tree_xyz,
    # unpack_widths,
    # cross2d,
)

import simpegtorch.discretize as discretize

torch.set_default_dtype(torch.float64)


TOL = 1e-8


@pytest.fixture
def vectors():
    return {
        "a": torch.tensor([1, 2, 3]),
        "b": torch.tensor([1, 2]),
        "c": torch.tensor([1, 2, 3, 4]),
    }


def test_mkvc1(vectors):
    x = mkvc(vectors["a"])
    assert x.shape == (3,)


def test_mkvc2(vectors):
    x = mkvc(vectors["a"], 2)
    assert x.shape == (3, 1)


def test_mkvc3(vectors):
    x = mkvc(vectors["a"], 3)
    assert x.shape == (3, 1, 1)


def test_ndgrid_3D():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2])
    c = torch.tensor([1, 2, 3, 4])

    XYZ = ndgrid([a, b, c])  # Make sure this returns a Fortran-stacked grid

    X1_test = torch.tensor(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    )
    X2_test = torch.tensor(
        [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
    )
    X3_test = torch.tensor(
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    )

    assert torch.all(XYZ[:, 0] == X1_test)
    assert torch.all(XYZ[:, 1] == X2_test)
    assert torch.all(XYZ[:, 2] == X3_test)


def test_sub2ind():
    x = torch.ones((5, 2))
    assert torch.all(sub2ind(x.shape, [0, 0]) == torch.tensor([0]))
    assert torch.all(sub2ind(x.shape, [4, 0]) == torch.tensor([4]))
    assert torch.all(sub2ind(x.shape, [0, 1]) == torch.tensor([5]))
    assert torch.all(sub2ind(x.shape, [4, 1]) == torch.tensor([9]))
    assert torch.all(sub2ind(x.shape, [[4, 1]]) == torch.tensor([9]))
    assert torch.all(
        sub2ind(x.shape, [[0, 0], [4, 0], [0, 1], [4, 1]]) == torch.tensor([0, 4, 5, 9])
    )


def test_ind2sub():
    x = torch.ones((5, 2))
    i, j = ind2sub(x.shape, [0, 4, 5, 9])
    assert torch.all(i == torch.tensor([0, 4, 0, 4]))
    assert torch.all(j == torch.tensor([0, 0, 1, 1]))


def test_is_scalar():
    assert is_scalar(1.0)
    assert is_scalar(1)
    assert is_scalar(1j)
    assert is_scalar(np.array(1.0))
    assert is_scalar(np.array(1))
    assert is_scalar(np.array(1j))
    assert is_scalar(np.array([1.0]))
    assert is_scalar(np.array([1]))
    assert is_scalar(np.array([1j]))
    assert is_scalar(np.array([[1.0]]))
    assert is_scalar(np.array([[1]]))
    assert is_scalar(np.array([[1j]]))

    assert is_scalar(torch.tensor(1.0))
    assert is_scalar(torch.tensor(1))
    assert is_scalar(torch.tensor([1.0]))
    assert is_scalar(torch.tensor([[1.0]]))


def test_as_array_n_by_dim():
    true = torch.tensor([[1.0, 2.0, 3.0]])

    list_array = as_array_n_by_dim([1, 2, 3], 3)
    assert torch.allclose(true - list_array, torch.zeros(len(true)))
    assert true.shape == list_array.shape

    list_array = as_array_n_by_dim(torch.tensor([1, 2, 3]), 3)
    assert torch.allclose(true - list_array, torch.zeros(len(true)))
    assert true.shape == list_array.shape

    list_array = as_array_n_by_dim(torch.tensor([[1, 2, 3.0]]), 3)
    assert torch.allclose(true - list_array, torch.zeros(len(true)))
    assert true.shape == list_array.shape

    true = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    list_array = as_array_n_by_dim([[1, 2], [4, 5]], 2)
    assert torch.allclose(true - list_array, torch.zeros(len(true)))
    assert true.shape == list_array.shape


def test_zero(vectors):
    z = Zero()
    assert z == 0
    assert not (z < 0)
    assert z <= 0
    assert not (z > 0)
    assert z >= 0
    assert +z == z
    assert -z == z
    assert z + 1 == 1
    assert z + 3 + z == 3
    assert z - 3 == -3
    assert z - 3 - z == -3
    assert 3 * z == 0
    assert z * 3 == 0
    assert z / 3 == 0

    a = 1
    a += z
    assert a == 1
    a = 1
    a += z
    assert a == 1

    with pytest.raises(ZeroDivisionError):
        3 / z

    assert mkvc(z) == 0
    assert sdiag(z) * a == 0
    assert z.T == 0
    assert z.transpose() == 0


def test_mat_zero():
    z = Zero()
    S = sdiag(torch.tensor([2, 3]))
    assert S * z == 0


def test_numpy_multiply():
    z = Zero()
    x = torch.tensor([1, 2, 3])
    a = x * z
    assert isinstance(a, Zero)

    z = Zero()
    x = torch.tensor([1, 2, 3])
    a = z * x
    assert isinstance(a, Zero)


def test_one():
    o = Identity()
    assert o == 1
    assert not (o < 1)
    assert o <= 1
    assert not (o > 1)
    assert o >= 1
    o = -o
    assert o == -1
    assert not (o < -1)
    assert o <= -1
    assert not (o > -1)
    assert o >= -1
    assert -1.0 * (-o) * o == -o
    o = Identity()
    assert +o == o
    assert -o == -o
    assert o * 3 == 3
    assert -o * 3 == -3
    assert -o * o == -1
    assert -o * o * -o == 1
    assert -o + 3 == 2
    assert 3 + -o == 2

    assert -o - 3 == -4
    assert o - 3 == -2
    assert 3 - -o == 4
    assert 3 - o == 2

    assert o // 2 == 0
    assert o / 2.0 == 0.5
    assert -o // 2 == -1
    assert -o / 2.0 == -0.5
    assert 2 / o == 2
    assert 2 // -o == -2
    assert 2.3 // o == 2
    assert 2.3 // -o == -3

    assert o.T == 1
    assert o.transpose() == 1


def test_mat_one():
    o = Identity()
    S = sdiag(torch.tensor([2, 3]))

    def check(exp, ans):
        assert torch.all(exp.to_dense() == torch.tensor(ans))

    check(S * o, [[2, 0], [0, 3]])
    check(o * S, [[2, 0], [0, 3]])
    check(S * -o, [[-2, 0], [0, -3]])
    check(-o * S, [[-2, 0], [0, -3]])
    check(S / o, [[2, 0], [0, 3]])
    check(S / -o, [[-2, 0], [0, -3]])

    check(S + o, [[3, 0], [0, 4]])
    check(o + S, [[3, 0], [0, 4]])
    check(S - o, [[1, 0], [0, 2]])

    check(S + -o, [[1, 0], [0, 2]])
    check(-o + S, [[1, 0], [0, 2]])


def test_mat_shape():
    o = Identity()
    S = sdiag(torch.tensor([2, 3]))

    assert torch.all((S * o).to_dense() == S.to_dense())
    assert torch.all((S * -o).to_dense() == -S.to_dense())


def test_torch_one():
    o = Identity()
    n = torch.tensor([2.0, 3])

    assert torch.all(n + 1 == n + o)
    assert torch.all(1 + n == o + n)
    assert torch.all(n - 1 == n - o)
    assert torch.all(1 - n == o - n)
    assert torch.all(n / 1 == n / o)
    assert torch.all(n / -1 == n / -o)
    assert torch.all(1 / n == o / n)
    assert torch.all(-1 / n == -o / n)
    assert torch.all(n * 1 == n * o)
    assert torch.all(n * -1 == n * -o)
    assert torch.all(1 * n == o * n)
    assert torch.all(-1 * n == -o * n)


def test_both():
    z = Zero()
    o = Identity()
    assert o * z == 0
    assert o * z + o == 1
    assert o - z == 1


def test_invXXXBlockDiagonal():
    torch.manual_seed(78352)
    a = [torch.rand((5, 1)) for i in range(4)]

    B = inverse_2x2_block_diagonal(*a)

    A = torch.vstack(
        (
            torch.hstack((sdiag(a[0]), sdiag(a[1]))),
            torch.hstack((sdiag(a[2]), sdiag(a[3]))),
        )
    )

    Z2 = B @ A - speye(10)
    assert torch.norm(Z2.to_dense().flatten(), 2) < TOL

    a = [torch.rand((5, 1)) for i in range(9)]
    B = inverse_3x3_block_diagonal(*a)

    A = torch.vstack(
        (
            torch.hstack((sdiag(a[0]), sdiag(a[1]), sdiag(a[2]))),
            torch.hstack((sdiag(a[3]), sdiag(a[4]), sdiag(a[5]))),
            torch.hstack((sdiag(a[6]), sdiag(a[7]), sdiag(a[8]))),
        )
    )

    Z3 = B @ A - speye(15)

    assert torch.norm(Z3.to_dense().flatten(), 2) < TOL


def test_TensorType2D():
    torch.manual_seed(8564)
    M = discretize.TensorMesh([6, 6])
    a1 = torch.rand(M.n_cells)
    a2 = torch.rand(M.n_cells)
    a3 = torch.rand(M.n_cells)
    prop1 = a1
    prop2 = torch.stack([a1, a2], dim=0).T
    prop3 = torch.stack([a1, a2, a3], dim=0).T

    for ii, prop in enumerate([4, prop1, prop2, prop3]):
        assert TensorType(M, prop) == ii

    with pytest.raises(Exception) as _:
        TensorType(M, np.c_[a1, a2, a3, a3])
    assert TensorType(M, None) == -1


def test_TensorType3D():
    torch.manual_seed(78352)
    M = discretize.TensorMesh([6, 6, 7])
    # Generate random tensors
    a1 = torch.rand(M.n_cells)
    a2 = torch.rand(M.n_cells)
    a3 = torch.rand(M.n_cells)
    a4 = torch.rand(M.n_cells)
    a5 = torch.rand(M.n_cells)
    a6 = torch.rand(M.n_cells)
    prop1 = a1

    # Stack along a new dimension and transpose
    prop2 = torch.stack([a1, a2, a3], dim=0).T
    prop3 = torch.stack([a1, a2, a3, a4, a5, a6], dim=0).T

    for ii, prop in enumerate([4, prop1, prop2, prop3]):
        assert TensorType(M, prop) == ii

    with pytest.raises(Exception) as _:
        TensorType(M, torch.stack([a1, a2, a3, a3], dim=0).T)

    assert TensorType(M, None) == -1


def test_inverse_property_tensor3D():
    # Use torch's random generator
    torch.manual_seed(78352)

    M = discretize.TensorMesh([6, 6, 6])

    # Generate random tensors
    a1 = torch.rand(M.n_cells)
    a2 = torch.rand(M.n_cells)
    a3 = torch.rand(M.n_cells)
    a4 = torch.rand(M.n_cells)
    a5 = torch.rand(M.n_cells)
    a6 = torch.rand(M.n_cells)

    prop1 = a1

    # Stack along a new dimension and transpose
    prop2 = torch.stack([a1, a2, a3], dim=0).T
    prop3 = torch.stack([a1, a2, a3, a4, a5, a6], dim=0).T

    for i, prop in enumerate([4, prop1, prop2, prop3]):
        b = inverse_property_tensor(M, prop)
        A = make_property_tensor(M, prop)
        B1 = make_property_tensor(M, b)
        B2 = inverse_property_tensor(M, prop, return_matrix=True)

        # Create identity tensor
        identity = speye(M.n_cells * 3)

        # Matrix multiplication and norm calculation
        Z = B1 @ A - identity

        # If Z is a sparse tensor, convert to dense for norm calculation
        if hasattr(Z, "to_dense"):
            Z_dense = Z.to_dense()
        else:
            Z_dense = Z

        assert (
            torch.norm(Z_dense.flatten(), 2) < TOL
        ), f"Norm of Z is not close to zero on iter: {i}"

        Z = B2 @ A - identity

        if hasattr(Z, "to_dense"):
            Z_dense = Z.to_dense()
        else:
            Z_dense = Z

        assert torch.norm(Z_dense.flatten(), 2) < TOL


def test_reshape_fortran():
    """Test the reshape_fortran function using NumPy's Fortran order as reference."""
    torch.manual_seed(8564)

    # Test 1: Basic 2D to 1D reshape
    x_np = np.arange(12).reshape(3, 4)
    x_torch = torch.from_numpy(x_np.copy())

    reshaped_torch = reshape_fortran(x_torch, (12,))
    expected_np = x_np.reshape(12, order="F")
    expected_torch = torch.from_numpy(expected_np)

    assert torch.equal(
        reshaped_torch, expected_torch
    ), "2D to 1D reshape failed against NumPy reference"

    # Test 2: 1D to 2D reshape
    x_np = np.arange(12)
    x_torch = torch.from_numpy(x_np.copy())

    reshaped_torch = reshape_fortran(x_torch, (3, 4))
    expected_np = x_np.reshape(3, 4, order="F")
    expected_torch = torch.from_numpy(expected_np)

    assert torch.equal(
        reshaped_torch, expected_torch
    ), "1D to 2D reshape failed against NumPy reference"

    # Test 3: 2D to 2D reshape (different dimensions)
    x_np = np.arange(12).reshape(2, 6)
    x_torch = torch.from_numpy(x_np.copy())

    reshaped_torch = reshape_fortran(x_torch, (4, 3))
    expected_np = x_np.reshape(4, 3, order="F")
    expected_torch = torch.from_numpy(expected_np)


def test_kron():
    """Test the kron function for Kronecker product of sparse matrices."""
    torch.manual_seed(8564)

    A_indices = torch.tensor([[0, 1], [0, 1]])
    A_values = torch.rand(2)
    A = torch.sparse_coo_tensor(A_indices, A_values, (2, 2))

    B_indices = torch.tensor([[0, 1], [0, 1]])
    B_values = torch.rand(2)
    B = torch.sparse_coo_tensor(B_indices, B_values, (2, 2))

    result = kron(A, B)
    assert result.shape == (4, 4), "Kronecker product dimensions incorrect"

    result_dense = result.to_dense()
    A_dense = A.to_dense()
    B_dense = B.to_dense()
    expected_dense = torch.kron(A_dense, B_dense)
    assert torch.allclose(
        result_dense, expected_dense
    ), "Kronecker product values incorrect"
