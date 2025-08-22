"""
PyTorch-based inversion framework for SimPEG-Torch

This module provides PyTorch implementations of SimPEG's inversion components,
leveraging automatic differentiation and GPU acceleration.
"""

import torch
import numpy as np
from typing import List, Optional, Union


class BaseInversion:
    """
    PyTorch-based inversion orchestrator.

    Manages the optimization loop, directives, and convergence for geophysical
    inversions using PyTorch's automatic differentiation capabilities.

    Parameters
    ----------
    inv_prob : BaseInvProblem
        The inverse problem containing data misfit, regularization, and optimization setup
    directives : list of InversionDirective, optional
        List of directives to control inversion behavior (beta cooling, convergence, etc.)
    device : str, optional
        PyTorch device ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float64)
    """

    def __init__(
        self,
        inv_prob,
        directives: Optional[List] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.inv_prob = inv_prob
        self.directives = directives or []
        self.device = device
        self.dtype = dtype

        # Set references for directives
        for directive in self.directives:
            directive.inversion = self
            directive.inv_prob = inv_prob
            directive.opt = inv_prob.opt if hasattr(inv_prob, "opt") else None

        # Inversion state
        self.model = None
        self.iteration = 0
        self.phi_d_history = []
        self.phi_m_history = []
        self.phi_history = []
        self.beta_history = []
        self.converged = False
        self.reason_for_stop = None

    def run(self, m0: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Run the inversion starting from initial model m0.

        Parameters
        ----------
        m0 : array_like
            Starting model

        Returns
        -------
        numpy.ndarray
            Recovered model
        """
        # Use the provided model directly (it should already be properly configured)
        if isinstance(m0, np.ndarray):
            m0 = torch.tensor(
                m0, dtype=self.dtype, device=self.device, requires_grad=True
            )
        elif isinstance(m0, torch.Tensor):
            # Don't modify the original tensor - it should already be set up correctly
            m0 = m0.to(dtype=self.dtype, device=self.device)

        self.model = m0

        # Initialize directives
        self._call_directives("initialize")

        # Setup optimizer
        optimizer = self.inv_prob.optimizer

        print(f"Running inversion with {len(self.directives)} directives")
        print(f"Initial model shape: {self.model.shape}")
        print(f"Device: {self.device}, dtype: {self.dtype}")

        # Main optimization loop
        while not self.converged and self.iteration < self.inv_prob.max_iter:

            # Zero gradients
            optimizer.zero_grad()

            # Compute objective function
            phi = self.inv_prob(self.model)

            # Backward pass for gradients
            phi.backward(retain_graph=True)

            # Store objective function components
            with torch.no_grad():
                phi_d = self.inv_prob.dmisfit(self.model)
                phi_m = self.inv_prob.reg(self.model)

                self.phi_d_history.append(phi_d.item())
                self.phi_m_history.append(phi_m.item())
                self.phi_history.append(phi.item())
                self.beta_history.append(self.inv_prob.beta)

            # Check gradients before step
            grad_norm = (
                self.model.grad.norm().item() if self.model.grad is not None else 0.0
            )

            # Update parameters
            optimizer.step()

            # Print iteration info
            if self.iteration % 1 == 0:
                print(
                    f"Iter {self.iteration:3d}: φ = {phi.item():.2e} "
                    f"(φ_d = {phi_d.item():.2e}, β×φ_m = {self.inv_prob.beta*phi_m.item():.2e}) "
                    f"grad_norm = {grad_norm:.2e}"
                )

            self.iteration += 1

            # Apply end-of-iteration directives
            self._call_directives("endIter")

        # Apply finishing directives
        self._call_directives("finish")

        print(f"Inversion completed after {self.iteration} iterations")
        if self.reason_for_stop:
            print(f"Reason for stopping: {self.reason_for_stop}")

        return self.model.detach().cpu().numpy()

    def _call_directives(self, directive_type: str):
        """Call all directives of specified type"""
        for directive in self.directives:
            if hasattr(directive, directive_type):
                getattr(directive, directive_type)()


class BaseInvProblem:
    """
    PyTorch inverse problem combining data misfit, regularization, and optimization.

    The inverse problem defines the objective function:
    φ(m) = φ_d(m) + β × φ_m(m)

    where φ_d is data misfit, φ_m is regularization, and β is trade-off parameter.

    Parameters
    ----------
    dmisfit : BaseDataMisfit
        Data misfit term
    reg : BaseRegularization
        Regularization term
    optimizer_class : str or torch.optim.Optimizer, optional
        PyTorch optimizer class name or class (default: 'Adam')
    optimizer_kwargs : dict, optional
        Keyword arguments for optimizer
    beta : float, optional
        Trade-off parameter between data misfit and regularization
    max_iter : int, optional
        Maximum number of iterations
    """

    def __init__(
        self,
        dmisfit,
        reg,
        optimizer: torch.optim.Optimizer,
        beta: float = 1.0,
        max_iter: int = 50,
    ):
        self.dmisfit = dmisfit
        self.reg = reg
        self.beta = beta
        self.max_iter = max_iter
        self.optimizer = optimizer

    def __call__(self, model: torch.Tensor) -> torch.Tensor:
        """
        Evaluate objective function: φ(m) = φ_d(m) + β × φ_m(m)

        Parameters
        ----------
        model : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            Objective function value
        """
        phi_d = self.dmisfit(model)
        phi_m = self.reg(model)
        return phi_d + self.beta * phi_m


class InversionDirective:
    """
    Base class for inversion directives.

    Directives control inversion behavior such as beta cooling schedules,
    convergence criteria, parameter updates, etc.
    """

    def __init__(self):
        self.inversion = None
        self.inv_prob = None
        self.opt = None

    def initialize(self):
        """Called at start of inversion"""
        pass

    def endIter(self):
        """Called at end of each iteration"""
        pass

    def finish(self):
        """Called at end of inversion"""
        pass


class BetaSchedule(InversionDirective):
    """
    Beta cooling schedule directive.

    Reduces the trade-off parameter β at regular intervals during inversion
    using: β_new = β_old / cooling_factor

    Parameters
    ----------
    cooling_factor : float, optional
        Factor by which beta is reduced (default: 8.0)
    cooling_rate : int, optional
        Number of iterations between beta reductions (default: 3)
    """

    def __init__(self, cooling_factor: float = 8.0, cooling_rate: int = 3):
        super().__init__()
        self.cooling_factor = cooling_factor
        self.cooling_rate = cooling_rate

    def endIter(self):
        """Apply beta cooling at specified intervals"""
        if (
            self.inversion.iteration > 0
            and self.inversion.iteration % self.cooling_rate == 0
        ):

            old_beta = self.inv_prob.beta
            self.inv_prob.beta /= self.cooling_factor

            print(
                f"BetaSchedule: β reduced from {old_beta:.2e} to {self.inv_prob.beta:.2e}"
            )


class TargetMisfit(InversionDirective):
    """
    Target misfit stopping criterion.

    Stops inversion when data misfit reaches target value, typically
    the expected chi-squared misfit for the noise level.

    Parameters
    ----------
    chi_factor : float, optional
        Target misfit as multiple of expected chi-squared (default: 1.0)
    """

    def __init__(self, chi_factor: float = 1.0):
        super().__init__()
        self.chi_factor = chi_factor
        self.target = None

    def initialize(self):
        """Set target misfit based on data"""
        # Assume target is number of data points for chi-squared = 1
        if hasattr(self.inv_prob.dmisfit, "n_data"):
            self.target = self.chi_factor * self.inv_prob.dmisfit.n_data
        else:
            # Fallback - use current misfit divided by large factor
            with torch.no_grad():
                current_misfit = self.inv_prob.dmisfit(self.inversion.model)
                self.target = current_misfit.item() / 100.0

        print(f"TargetMisfit: target = {self.target:.2e}")

    def endIter(self):
        """Check if target misfit reached"""
        if self.target is not None:
            current_misfit = self.inversion.phi_d_history[-1]
            if current_misfit <= self.target:
                self.inversion.converged = True
                self.inversion.reason_for_stop = (
                    f"Target misfit reached: {current_misfit:.2e} <= {self.target:.2e}"
                )


class BetaEstimate_ByEig(InversionDirective):
    """
    Estimate initial beta using eigenvalue method.

    Sets β such that data misfit and regularization terms have similar magnitudes
    initially, using the ratio of their dominant eigenvalues.

    Parameters
    ----------
    beta0_ratio : float, optional
        Ratio for setting initial beta (default: 1.0)
    """

    def __init__(self, beta0_ratio: float = 1.0):
        super().__init__()
        self.beta0_ratio = beta0_ratio

    def initialize(self):
        """Estimate and set initial beta"""
        # Simple heuristic: ratio of current misfit values
        with torch.no_grad():
            phi_d = self.inv_prob.dmisfit(self.inversion.model)
            phi_m = self.inv_prob.reg(self.inversion.model)

            if phi_m.item() > 0:
                beta_est = self.beta0_ratio * phi_d.item() / phi_m.item()
                self.inv_prob.beta = beta_est
                print(
                    f"BetaEstimate: β set to {beta_est:.2e} (ratio = {self.beta0_ratio})"
                )
            else:
                print("BetaEstimate: regularization is zero, keeping default beta")
