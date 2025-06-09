"""
molcollisions: Hash Collisions in Molecular Fingerprints

A research package for studying the effects of fingerprint compression
on Gaussian Process performance in molecular property prediction.

Example usage:
    >>> from molcollisions import SparseFingerprint, FingerprintConfig
    >>> config = FingerprintConfig(name='test', size='sparse')
    >>> fp = SparseFingerprint(config)
    >>> fingerprint = fp('CCO')  # ethanol
"""

__version__ = "0.1.0"
__author__ = "Walter Virany"

from .fingerprints import (
    SparseFingerprint,
    CompressedFingerprint, 
    FingerprintConfig,
    create_fingerprint
)

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SparseFingerprint",
    "CompressedFingerprint",
    "FingerprintConfig", 
    "create_fingerprint"
]