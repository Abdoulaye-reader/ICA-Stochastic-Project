"""
ICA - Independent Component Analysis algorithms.

Algorithms implemented:
- FastICA  : fixed-point algorithm (deflation and symmetric modes)
- Infomax  : Bell-Sejnowski gradient ascent on log-likelihood
- JADE     : Joint Approximate Diagonalization of Eigenmatrices
"""

from .fastica import FastICA
from .infomax import InfomaxICA
from .jade import JADE
from .utils import (
    whiten,
    amari_error,
    generate_sources,
    mix_sources,
)

__all__ = [
    "FastICA",
    "InfomaxICA",
    "JADE",
    "whiten",
    "amari_error",
    "generate_sources",
    "mix_sources",
]
