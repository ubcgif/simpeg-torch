import torch
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.resistivity import (
    DC3DCellCentered,
    SrcDipole,
    RxDipole,
    Survey,
)

from simpegtorch.simulation.base import DirectSolver, mappings

from simpegtorch.discretize.utils import (
    ndgrid,
)

# Create a tensor mesh
hx = torch.ones(10) * 25
hy = torch.ones(10) * 25
hz = torch.ones(10) * 25

# 250m x 250m x 250m mesh

# Origin at (-125, -125, -250) to center the mesh
origin = torch.tensor([-125.0, -125.0, -250.0])

mesh = TensorMesh(
    [hx, hy, hz],
    origin=origin,
)

sigma = torch.ones(mesh.nC) * 1e-2  # Uniform conductivity
sigma_map = mappings.BaseMapping(sigma)

# Set up survey parameters for numeric solution
x = mesh.cell_centers_x[(mesh.cell_centers_x > -75.0) & (mesh.cell_centers_x < 75.0)]
y = mesh.cell_centers_y[(mesh.cell_centers_y > -75.0) & (mesh.cell_centers_y < 75.0)]

M = ndgrid(x - 25.0, y, [0.0])
N = ndgrid(x + 25.0, y, [0.0])

# create a dipole dipole survey
rx = RxDipole(
    locations_m=M,
    locations_n=N,
)

loc_a = torch.tensor([-25.0, 0.0, 0.0])
loc_b = torch.tensor([25.0, 0.0, 0.0])

src = SrcDipole(
    [rx],
    loc_a,  # location of A
    loc_b,  # location of B
    current=1.0,  # current in Amperes
)

survey = Survey([src])

## Setup the Problem as a Cell Centered PDE problem
problem = DC3DCellCentered(
    mesh,
    survey,
    sigma_map,
    bc_type="Dirichlet",
)

# Create the solver we will use
solver = DirectSolver(problem)

# Solve the forward problem
Data = solver.forward()

# Print results
print("Forward simulation completed successfully!")
print(f"Data values: {Data}")
print(f"Data min: {Data.min():.6e}, max: {Data.max():.6e}")
