import torch 
from discretize.utils import (
    mkvc,
    Zero
)

class BaseFDEMSrc():
    """
    Base FDEN source class 
    ----------
    receiver_list : list of simpeg.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    location : (dim) numpy.ndarray, default: ``None``
        Source location.
    """

    
    _ePrimary = None
    _bPrimary = None
    _hPrimary = None
    _jPrimary = None

    def __init__(self, receiver_list, frequency, location=None, **kwargs):
        super().__init__(receiver_list=receiver_list, location=location, **kwargs)
        self.frequency = frequency

    @property
    def frequency(self):
        """Source frequency

        Returns
        -------
        float
            Source frequency
        """
        return self._frequency

    @frequency.setter
    def frequency(self, freq):
        freq = freq
        self._frequency = freq

    def bPrimary(self, simulation):
        """Compute primary magnetic flux density

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic flux density
        """
        if self._bPrimary is None:
            return Zero()
        return self._bPrimary

    def bPrimaryDeriv(self, simulation, v, adjoint=False):
        """Compute derivative of primary magnetic flux density times a vector

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of primary magnetic flux density times a vector
        """
        return Zero()

    def hPrimary(self, simulation):
        """Compute primary magnetic field

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic field
        """
        if self._hPrimary is None:
            return Zero()
        return self._hPrimary

    def ePrimary(self, simulation):
        """Compute primary electric field

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary electric field
        """
        if self._ePrimary is None:
            return Zero()
        return self._ePrimary

    def jPrimary(self, simulation):
        """Compute primary current density

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary current density
        """
        if self._jPrimary is None:
            return Zero()
        return self._jPrimary



