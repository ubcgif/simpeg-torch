import torch
from simpegtorch.discretize.tests import OrderTest

MESHTYPES = ["uniformTensorMesh", "randomTensorMesh"]
TOLERANCES = [0.9, 0.5, 0.5]
TOL = 1e-7


def call1(fun, xyz):
    return fun(xyz)


def call2(fun, xyz):
    fun(xyz[:, 0], xyz[:, -1])


def call3(fun, xyz):
    fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])


def cart_row2(g, xfun, yfun):
    return torch.stack([call2(xfun, g), call2(yfun, g)], dim=1)


def cart_row3(g, xfun, yfun, zfun):
    return torch.stack([call3(xfun, g), call3(yfun, g), call3(zfun, g)], dim=1)


def cartF2(M, fx, fy):
    return torch.cat([cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)], dim=0)


def cartF2Cyl(M, fx, fy):
    return torch.cat([cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFz, fx, fy)], dim=0)


def cartE2(M, ex, ey):
    return torch.cat([cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)], dim=0)


def cartE2Cyl(M, ex, ey):
    return cart_row2(M.gridEy, ex, ey)


def cartF3(M, fx, fy, fz):
    return torch.cat(
        [
            cart_row3(M.gridFx, fx, fy, fz),
            cart_row3(M.gridFy, fx, fy, fz),
            cart_row3(M.gridFz, fx, fy, fz),
        ],
        dim=0,
    )


def cartE3(M, ex, ey, ez):
    return torch.cat(
        [
            cart_row3(M.gridEx, ex, ey, ez),
            cart_row3(M.gridEy, ex, ey, ez),
            cart_row3(M.gridEz, ex, ey, ez),
        ],
        dim=0,
    )


class TestInterpolation1D(OrderTest):
    name = "Interpolation 1D"
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 1
    meshSizes = [8, 16, 32, 64, 128]

    generator = torch.Generator().manual_seed(55124)
    LOCS = torch.rand(50, generator=generator) * 0.6 + 0.2

    def getError(self):
        def funX(x):
            torch.cos(2 * torch.pi * x)

        ana = call1(funX, self.LOCS)

        if "CC" == self.type:
            grid = call1(funX, self.M.gridCC)
        elif "N" == self.type:
            grid = call1(funX, self.M.gridN)

        comp = self.M.get_interpolation_matrix(self.LOCS, self.type) * grid

        err = torch.linalg.norm((comp - ana), 2)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 1D: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.name = "Interpolation 1D: N"
        self.orderTest()
