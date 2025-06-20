import torch
import numpy as np
import pytest

from simpegtorch.discretize.utils import (
    sdiag,
    sdinv,
    sub2ind,
    speye,
    av,
    av_extrap,
    ddx,
    ndgrid,
    mkvc,
    kron,
    reshape_fortran,
    is_scalar,
    inverse_2x2_block_diagonal,
    inverse_3x3_block_diagonal,
    inverse_property_tensor,
    make_property_tensor,
    get_diag,
    make_boundary_bool,
    # index_cube,
    ind2sub,
    as_array_n_by_dim,
    TensorType,
    Zero,
    Identity,
    torch_blockdiag,
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


# Device fixtures for testing across different devices
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    """Pytest fixture to test on CPU and CUDA (if available)."""
    return torch.device(request.param)


@pytest.fixture
def vectors(device):
    return {
        "a": torch.tensor([1, 2, 3], device=device),
        "b": torch.tensor([1, 2], device=device),
        "c": torch.tensor([1, 2, 3, 4], device=device),
    }


def test_mkvc1(vectors, device):
    x = mkvc(vectors["a"])
    assert x.shape == (3,)
    assert x.device.type == device.type


def test_mkvc2(vectors, device):
    x = mkvc(vectors["a"], 2)
    assert x.shape == (3, 1)
    assert x.device.type == device.type


def test_mkvc3(vectors):
    x = mkvc(vectors["a"], 3)
    assert x.shape == (3, 1, 1)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_ndgrid_3D(device):
    device = torch.device(device)
    a = torch.tensor([1, 2, 3], device=device)
    b = torch.tensor([1, 2], device=device)
    c = torch.tensor([1, 2, 3, 4], device=device)

    XYZ = ndgrid(
        [a, b, c], device=device
    )  # Make sure this returns a Fortran-stacked grid

    X1_test = torch.tensor(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        device=device,
    )
    X2_test = torch.tensor(
        [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
        device=device,
    )
    X3_test = torch.tensor(
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
        device=device,
    )

    assert torch.all(XYZ[:, 0] == X1_test)
    assert torch.all(XYZ[:, 1] == X2_test)
    assert torch.all(XYZ[:, 2] == X3_test)
    assert XYZ.device.type == device.type


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
        assert torch.all(exp.to_dense() == torch.tensor(ans)), f"{exp} != {ans}"

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


def test_ddx():
    ddx_mat = ddx(3)
    assert ddx_mat.shape == (3, 4), "ddx matrix shape is incorrect"
    ddx_mat_dense = ddx_mat.to_dense()

    expected_ddx = torch.tensor(
        [
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(
        ddx_mat_dense, expected_ddx
    ), "ddx matrix values are incorrect"


def test_av():
    av_mat = av(3)
    assert av_mat.shape == (3, 4), "av matrix shape is incorrect"
    av_mat_dense = av_mat.to_dense()
    expected_av = torch.tensor(
        [
            [0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5],
        ]
    )
    assert torch.allclose(av_mat_dense, expected_av), "av matrix values are incorrect"


def test_av_extrap():
    av_mat = av_extrap(3)
    assert av_mat.shape == (4, 3), "av matrix shape is incorrect"
    av_mat_dense = av_mat.to_dense()
    expected_av = torch.tensor(
        [
            [1, 0, 0],
            [0.5, 0.5, 0.0],
            [0, 0.5, 0.5],
            [0, 0, 1],
        ]
    )
    assert torch.allclose(av_mat_dense, expected_av), "av matrix values are incorrect"


def test_get_diag_identity():
    # Test with identity matrix
    I = speye(4)
    diag = get_diag(I)
    expected = torch.ones(4, dtype=torch.float64)
    assert torch.allclose(diag, expected), "get_diag failed for identity matrix"


def test_get_diag_diagonal():
    # Test with arbitrary diagonal matrix
    diag_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    S = sdiag(diag_values)
    extracted_diag = get_diag(S)
    assert torch.allclose(
        extracted_diag, diag_values
    ), "get_diag failed for diagonal matrix"


def test_get_diag_sparse_with_zeros():
    # Test with sparse matrix having zeros on diagonal
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    values = torch.tensor([1.0, 0.0, 3.0])
    S = torch.sparse_coo_tensor(indices, values, (3, 3))

    diag = get_diag(S)
    expected = torch.tensor([1.0, 0.0, 3.0])
    assert torch.allclose(diag, expected), "get_diag failed with zeros on diagonal"


def test_get_diag_csr_format():
    # Test with CSR format
    diag_values = torch.tensor([2.0, -1.0, 5.0])
    S = sdiag(diag_values, sparse_type="csr")
    extracted_diag = get_diag(S)
    assert torch.allclose(extracted_diag, diag_values), "get_diag failed for CSR format"


def test_get_diag_empty_matrix():
    # Test with matrix that has missing diagonal elements
    indices = torch.tensor([[0, 2], [0, 2]])  # Missing diagonal at (1,1)
    values = torch.tensor([1.0, 3.0])
    S = torch.sparse_coo_tensor(indices, values, (3, 3))

    diag = get_diag(S)
    expected = torch.tensor([1.0, 0.0, 3.0])  # Should fill missing with 0
    assert torch.allclose(
        diag, expected
    ), "get_diag failed with missing diagonal elements"


def test_make_boundary_bool_1d():
    # Test 1D case
    shape = (5,)

    # Test all boundaries (x only for 1D)
    bool_ind = make_boundary_bool(shape, bdir="x")
    expected = torch.tensor([True, False, False, False, True])
    assert torch.equal(bool_ind, expected), "1D boundary bool failed for x direction"

    # Test default (should be same as "xyz" but only x applies)
    bool_ind_default = make_boundary_bool(shape)
    assert torch.equal(
        bool_ind_default, expected
    ), "1D boundary bool failed for default"


def test_make_boundary_bool_2d():
    # Test 2D case - matches the original discretize example
    shape = (3, 3)

    # Test all boundaries (corrected expected values)
    bool_ind = make_boundary_bool(shape)
    expected = torch.tensor([True, True, True, True, False, True, True, True, True])
    assert torch.equal(bool_ind, expected), "2D boundary bool failed for all directions"

    # Test x boundaries only
    bool_ind_x = make_boundary_bool(shape, bdir="x")
    expected_x = torch.tensor([True, False, True, True, False, True, True, False, True])
    assert torch.equal(
        bool_ind_x, expected_x
    ), "2D boundary bool failed for x direction"

    # Test y boundaries only
    bool_ind_y = make_boundary_bool(shape, bdir="y")
    expected_y = torch.tensor([True, True, True, False, False, False, True, True, True])
    assert torch.equal(
        bool_ind_y, expected_y
    ), "2D boundary bool failed for y direction"


def test_make_boundary_bool_3d():
    # Test 3D case with larger shape to have interior points
    shape = (3, 3, 3)

    # Test all boundaries
    bool_ind = make_boundary_bool(shape)
    # Center point at (1,1,1) should not be a boundary
    assert (
        bool(bool_ind[13]) is False
    ), "3D center point should not be boundary"  # Index 13 is center in Fortran order

    # Test z boundaries only
    bool_ind_z = make_boundary_bool(shape, bdir="z")
    # Should mark z=0 and z=2 planes (18 points total)
    assert torch.sum(bool_ind_z).item() == 18, "3D z boundary count should be 18"

    # For 2x2x2 case, all points are boundaries
    shape_small = (2, 2, 2)
    bool_ind_small = make_boundary_bool(shape_small)
    expected_small = torch.ones(8, dtype=torch.bool)
    assert torch.equal(
        bool_ind_small, expected_small
    ), "2x2x2 cube should have all boundary points"


def test_make_boundary_bool_larger_3d():
    # Test larger 3D case to better validate boundaries
    shape = (3, 3, 3)

    # Test x boundaries only
    bool_ind_x = make_boundary_bool(shape, bdir="x")
    # Should mark x=0 and x=2 planes
    expected_count_x = 2 * 3 * 3  # Two 3x3 planes
    assert (
        torch.sum(bool_ind_x).item() == expected_count_x
    ), "3D x boundary count incorrect"

    # Test combined boundaries
    bool_ind_xy = make_boundary_bool(shape, bdir="xy")
    # Should have more boundary points than just x
    assert torch.sum(bool_ind_xy) > torch.sum(
        bool_ind_x
    ), "3D xy boundary should be larger than x only"


def test_make_boundary_bool_no_boundaries():
    # Test with empty bdir string
    shape = (3, 3)
    bool_ind = make_boundary_bool(shape, bdir="")
    expected = torch.zeros(9, dtype=torch.bool)
    assert torch.equal(bool_ind, expected), "Empty bdir should return all False"


def test_make_boundary_bool_partial_directions():
    # Test various combinations of directions
    shape = (3, 3, 3)

    # Test "xz" combination
    bool_ind_xz = make_boundary_bool(shape, bdir="xz")
    bool_ind_x = make_boundary_bool(shape, bdir="x")
    bool_ind_z = make_boundary_bool(shape, bdir="z")

    # xz should be the union of x and z boundaries
    expected_xz = bool_ind_x | bool_ind_z
    assert torch.equal(
        bool_ind_xz, expected_xz
    ), "xz boundary should be union of x and z"


def test_sdinv_basic():
    # Test basic diagonal matrix inversion
    diag_values = torch.tensor([2.0, 4.0, 0.5])
    S = sdiag(diag_values)
    S_inv = sdinv(S)

    # Check that S * S_inv = I
    product = S @ S_inv
    expected_diag = torch.ones(3)
    actual_diag = get_diag(product)
    assert torch.allclose(
        actual_diag, expected_diag
    ), "sdinv should produce correct inverse"


def test_sdinv_error_cases():
    # Test error when matrix has zeros on diagonal
    diag_values = torch.tensor([1.0, 0.0, 3.0])
    S = sdiag(diag_values)

    with pytest.raises(ZeroDivisionError):
        sdinv(S)


def test_torch_blockdiag_basic():
    """Test basic functionality of torch_blockdiag."""
    # Create simple diagonal matrices
    A = sdiag(torch.tensor([1.0, 2.0]))
    B = sdiag(torch.tensor([3.0, 4.0, 5.0]))

    result = torch_blockdiag([A, B])

    # Check shape
    assert result.shape == (5, 5), "Block diagonal shape should be (5, 5)"

    # Check structure by converting to dense
    result_dense = result.to_dense()
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
        ]
    )

    assert torch.allclose(result_dense, expected), "Block diagonal structure incorrect"


def test_torch_blockdiag_single_matrix():
    """Test torch_blockdiag with a single matrix."""
    A = sdiag(torch.tensor([1.0, 2.0, 3.0]))
    result = torch_blockdiag([A])

    # Should be identical to input matrix
    assert torch.allclose(result.to_dense(), A.to_dense()), "Single matrix case failed"


def test_torch_blockdiag_empty_input():
    """Test torch_blockdiag with empty input."""
    with pytest.raises(ValueError):
        torch_blockdiag([])


def test_torch_blockdiag_tuple_input():
    """Test torch_blockdiag with tuple input."""
    A = sdiag(torch.tensor([1.0, 2.0]))
    B = sdiag(torch.tensor([3.0, 4.0]))

    result = torch_blockdiag((A, B))
    expected = torch_blockdiag([A, B])

    assert torch.allclose(result.to_dense(), expected.to_dense()), "Tuple input failed"


def test_torch_blockdiag_csr_format():
    """Test torch_blockdiag with CSR output format."""
    A = sdiag(torch.tensor([1.0, 2.0]), sparse_type="csr")
    B = sdiag(torch.tensor([3.0, 4.0]), sparse_type="csr")

    result = torch_blockdiag([A, B], sparse_type="csr")

    assert result.is_sparse_csr, "Result should be in CSR format"
    assert result.shape == (4, 4), "CSR result shape incorrect"


def test_torch_blockdiag_mixed_sizes():
    """Test torch_blockdiag with matrices of different sizes."""
    # Create matrices with different shapes
    A = sdiag(torch.tensor([1.0]))  # 1x1
    B = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [0, 1, 1]]), torch.tensor([2.0, 3.0, 4.0]), (2, 2)
    )  # 2x2
    C = sdiag(torch.tensor([5.0, 6.0, 7.0]))  # 3x3

    result = torch_blockdiag([A, B, C])

    assert result.shape == (6, 6), "Mixed sizes shape incorrect"

    # Check specific values
    result_dense = result.to_dense()
    assert result_dense[0, 0] == 1.0, "A block incorrect"
    assert result_dense[1, 1] == 2.0, "B block position (0,0) incorrect"
    assert result_dense[1, 2] == 3.0, "B block position (0,1) incorrect"
    assert result_dense[2, 2] == 4.0, "B block position (1,1) incorrect"
    assert result_dense[3, 3] == 5.0, "C block position (0,0) incorrect"
    assert result_dense[4, 4] == 6.0, "C block position (1,1) incorrect"
    assert result_dense[5, 5] == 7.0, "C block position (2,2) incorrect"


def test_torch_blockdiag_empty_matrices():
    """Test torch_blockdiag with matrices that have no non-zero elements."""
    device = torch.device("cpu")

    # Create empty sparse matrices
    A = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long, device=device),
        torch.zeros(0, dtype=torch.float64, device=device),
        (2, 2),
        device=device,
    )
    B = sdiag(torch.tensor([1.0, 2.0]))

    result = torch_blockdiag([A, B])

    assert result.shape == (4, 4), "Shape with empty matrix incorrect"
    result_dense = result.to_dense()

    # A block should be all zeros
    assert torch.all(result_dense[:2, :2] == 0.0), "Empty matrix block should be zero"
    # B block should have diagonal values
    assert result_dense[2, 2] == 1.0, "Non-empty block incorrect"
    assert result_dense[3, 3] == 2.0, "Non-empty block incorrect"


def test_torch_blockdiag_error_cases():
    """Test torch_blockdiag error handling."""
    # Test with non-sparse matrices
    A = torch.tensor([[1.0, 0.0], [0.0, 2.0]])  # Dense matrix
    B = sdiag(torch.tensor([3.0, 4.0]))

    with pytest.raises(TypeError):
        torch_blockdiag([A, B])


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_torch_blockdiag_device_support(device):
    """Test torch_blockdiag on different devices."""
    device = torch.device(device)

    A = sdiag(torch.tensor([1.0, 2.0], device=device))
    B = sdiag(torch.tensor([3.0, 4.0, 5.0], device=device))

    result = torch_blockdiag([A, B])

    assert result.device.type == device.type, f"Result device should be {device}"
    assert result.shape == (5, 5), "Device test shape incorrect"
