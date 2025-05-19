import torch
import numpy as np
import pytest

from simpegtorch.discretize.utils import (
    sdiag,
    sub2ind,
    ndgrid,
    mkvc,
    is_scalar,
    # inverse_2x2_block_diagonal,
    # inverse_3x3_block_diagonal,
    # inverse_property_tensor,
    # make_property_tensor,
    # index_cube,
    ind2sub,
    as_array_n_by_dim,
    # TensorType,
    Zero,
    Identity,
    # extract_core_mesh,
    # active_from_xyz,
    # mesh_builder_xyz,
    # refine_tree_xyz,
    # unpack_widths,
    # cross2d,
)

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
