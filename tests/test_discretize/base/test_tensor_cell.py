import pytest
import torch
from simpegtorch.discretize import TensorCell
from simpegtorch.discretize.tensor_mesh import _slice_to_index


@pytest.mark.parametrize(
    "slice_indices, expected_result",
    [
        (slice(None, None, None), range(8)),
        (slice(0, None, None), range(8)),
        (slice(1, None, None), range(1, 8)),
        (slice(None, 4, None), range(4)),
        (slice(None, 8, None), range(8)),
        (slice(None, None, 1), range(8)),
        (slice(None, None, 2), range(0, 8, 2)),
        (slice(None, None, -1), reversed(range(0, 8, 1))),
        (slice(1, 7, -2), reversed(range(1, 7, 2))),
        (slice(1, -1, None), range(1, 7, 1)),
        (slice(1, -2, 2), range(1, 6, 2)),
    ],
)
def test_slice_to_index(slice_indices, expected_result):
    end = 8
    indices = tuple(i for i in _slice_to_index(slice_indices, end))
    expected_result = tuple(i for i in expected_result)
    assert indices == expected_result


class TestTensorCell:

    @pytest.fixture(params=("1D", "2D", "3D"))
    def cell(self, request):
        dim = request.param
        if dim == "1D":
            h = torch.tensor([4.0])
            origin = torch.tensor([-2.0])
            index_unraveled = (1,)
            mesh_shape = (8,)
        elif dim == "2D":
            h = torch.tensor([4.0, 2.0])
            origin = torch.tensor([-2.0, 5.0])
            index_unraveled = (1, 2)
            mesh_shape = (8, 3)
        elif dim == "3D":
            h = torch.tensor([4.0, 2.0, 10.0])
            origin = torch.tensor([-2.0, 5.0, -12.0])
            index_unraveled = (1, 2, 3)
            mesh_shape = (8, 3, 10)
        return TensorCell(h, origin, index_unraveled, mesh_shape)

    def test_center(self, cell):
        if cell.dim == 1:
            true_center = torch.tensor([0.0], dtype=torch.float64)
        elif cell.dim == 2:
            true_center = torch.tensor([0.0, 6.0], dtype=torch.float64)
        elif cell.dim == 3:
            true_center = torch.tensor([0.0, 6.0, -7.0], dtype=torch.float64)
        assert torch.allclose(cell.center, true_center)

    def test_index(self, cell):
        if cell.dim == 1:
            true_index = 1
        elif cell.dim == 2:
            true_index = 17
        elif cell.dim == 3:
            true_index = 89
        assert cell.index == true_index

    def test_index_unraveled(self, cell):
        if cell.dim == 1:
            true_index_unraveled = (1,)
        elif cell.dim == 2:
            true_index_unraveled = (1, 2)
        elif cell.dim == 3:
            true_index_unraveled = (1, 2, 3)
        assert cell.index_unraveled == true_index_unraveled

    def test_bounds(self, cell):
        if cell.dim == 1:
            true_bounds = torch.tensor([-2.0, 2.0], dtype=torch.float64)
        elif cell.dim == 2:
            true_bounds = torch.tensor([-2.0, 2.0, 5.0, 7.0], dtype=torch.float64)
        elif cell.dim == 3:
            true_bounds = torch.tensor(
                [-2.0, 2.0, 5.0, 7.0, -12.0, -2.0], dtype=torch.float64
            )
        assert torch.allclose(cell.bounds, true_bounds)

    @pytest.mark.parametrize("change_h", (True, False))
    @pytest.mark.parametrize("change_origin", (True, False))
    @pytest.mark.parametrize("change_index", (True, False))
    @pytest.mark.parametrize("change_mesh_shape", (True, False))
    def test_eq(self, cell, change_h, change_origin, change_index, change_mesh_shape):
        h, origin = cell.h.clone(), cell.origin.clone()
        index_unraveled, mesh_shape = cell.index_unraveled, cell.mesh_shape
        if change_h:
            h += 0.1
        if change_origin:
            origin += 0.1
        if change_index:
            index_unraveled = tuple(i - 1 for i in index_unraveled)
        if change_mesh_shape:
            mesh_shape = tuple(i + 1 for i in mesh_shape)
        other_cell = TensorCell(h, origin, index_unraveled, mesh_shape)
        if any((change_origin, change_h, change_index, change_mesh_shape)):
            assert cell != other_cell
        else:
            assert cell == other_cell

    def test_eq_invalid_type(self, cell):
        class Dummy:
            pass

        other_object = Dummy()
        msg = "Cannot compare an object of type 'Dummy'"
        with pytest.raises(TypeError, match=msg):
            assert cell == other_object
