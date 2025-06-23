from abc import ABC, abstractmethod
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

cache_size = 300_000


class BaseFP(ABC):
    def __init__(self, radius: int = 2, count: bool = True):
        self.radius = radius
        self.count = count

    @abstractmethod
    def __call__(self, smiles: str):
        """
        Convert SMILES string to fingerprint
        """
        pass

    def _validate_smiles(self, smiles: str) -> Chem.Mol:
        """
        Validate and convert SMILES to RDKit molecule
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        return mol


class SparseFP(BaseFP):
    """
    Sparse fingerprint implementation - no hash collisions.
    """

    def __init__(self, radius: int = 2, count: bool = True):
        """
        Initialize sparse fingerprint.

        Args:
            radius: Radius for fingerprint generation
            count: If True, use count fingerprint; otherwise, use binary
        """
        super().__init__(radius, count)

        # Create fingerprint generator
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    @lru_cache(maxsize=cache_size)  # Cache fingerprints for BO performance
    def __call__(self, smiles: str):
        mol = self._validate_smiles(smiles)

        if self.count:
            return self.fpgen.GetSparseCountFingerprint(mol)
        else:
            return self.fpgen.GetSparseFingerprint(mol)


class CompressedFP(BaseFP):
    """
    Compressed fingerprint implementation - fixed size with potential hash collisions
    """

    def __init__(self, radius: int = 2, fp_size: int = 2048, count: bool = True):
        """
        Iniitalize compressed fingerprint.

        Args:
            radius: Radius for fingerprint generation
            fp_size: Fixed size of fingerprint vector
            count: If True, use count fingerprint; otherwise, use binary
        """
        super().__init__(radius, count)
        self.fp_size = fp_size

        # Create fingerprint generator
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    @lru_cache(maxsize=cache_size)  # Cache fingerprints for BO performance
    def __call__(self, smiles: str):
        mol = self._validate_smiles(smiles)

        if self.count:
            return self.fpgen.GetCountFingerprint(mol)
        else:
            return self.fpgen.GetFingerprint(mol)
