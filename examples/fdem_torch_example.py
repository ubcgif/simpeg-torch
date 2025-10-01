import torch
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.FDEM import (
    FDEM3DMagneticFluxDensity,
    MagneticDipole,
    RxPointMagneticFluxDensity,
    Survey,
)

from simpegtorch.simulation.base import DirectSolver, mappings

# Create a tensor mesh
hx = torch.ones(16) * 25  # 16 cells of 25m
hy = torch.ones(16) * 25
hz = torch.ones(16) * 25

# 400m x 400m x 400m mesh
# Origin at (0, 0, 0)
origin = torch.tensor([0.0, 0.0, 0.0])

mesh = TensorMesh(
    [hx, hy, hz],
    origin=origin,
)

# Uniform conductivity model: 0.01 S/m (100 Ohm-m resistivity)
sigma = torch.ones(mesh.nC) * 0.01
sigma_map = mappings.BaseMapping(sigma)

# Set up FDEM survey
# Place receivers in a line to measure magnetic flux density
rx_locations = torch.tensor(
    [
        [100.0, 200.0, 0.0],  # Surface measurements
        [150.0, 200.0, 0.0],
        [200.0, 200.0, 0.0],
        [250.0, 200.0, 0.0],
        [300.0, 200.0, 0.0],
    ],
    dtype=torch.float64,
)

# Measure vertical component of magnetic flux density (Bz)
rx = RxPointMagneticFluxDensity(
    locations=rx_locations, orientation="z", component="real"
)

# Create a vertical magnetic dipole source
src_location = torch.tensor([200.0, 200.0, 50.0])  # 50m below surface
src_moment = torch.tensor([0.0, 0.0, 1.0])  # Vertical moment (Az = 1 A⋅m²)
frequency = 10.0  # 10 Hz

src = MagneticDipole(
    [rx],
    frequency,
    src_location,
    moment=src_moment,
)

survey = Survey([src])

## Setup the FDEM PDE problem
problem = FDEM3DMagneticFluxDensity(
    mesh,
    survey,
    sigma_map,
)

# Create the solver we will use
solver = DirectSolver(problem)

# Solve the forward problem
Data = solver.forward()

# Print results
print("FDEM Forward simulation completed successfully!")
print(f"Frequency: {frequency} Hz")
print(f"Source location: {src.location}")
print(f"Conductivity: {sigma[0]:.4f} S/m ({1/sigma[0]:.1f} Ohm-m)")
print(f"Number of receivers: {rx.locations.shape[0]}")
print(f"Data values (real part): {Data.real}")
print(f"Data min: {Data.real.min():.6e} T, max: {Data.real.max():.6e} T")
