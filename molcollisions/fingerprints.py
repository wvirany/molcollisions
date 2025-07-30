from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from sort_and_slice_ecfp_featuriser import create_sort_and_slice_ecfp_featuriser

cache_size = 300_000


class MolecularFingerprint(ABC):
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

    @abstractmethod
    def get_fp_type(self) -> str:
        pass


class SparseFP(MolecularFingerprint):
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

    def get_fp_type(self) -> str:
        return f"sparse-r{self.radius}"


class CompressedFP(MolecularFingerprint):
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

    def get_fp_type(self) -> str:
        return f"compressed{self.fp_size}-r{self.radius}"


class SortSliceFP(MolecularFingerprint):
    """
    Sort&Slice fingerprint implementation - uses substructure pooling as described in
    Dablander et al. (2024).

    Uses a training dataset (we are primarily using ZINC250K) to select the most
    prevalent substructures, then creates fixed-size fingerprints by mapping molecules
    to these learned features.
    """

    def __init__(
        self,
        dataset_path: Path = Path("data/zinc250k.smiles"),
        radius: int = 2,
        fp_size: int = 2048,
        count: bool = True,
    ):
        """
        Initialize Sort&Slice fingerprint.

        Args:
            dataset_path: Path to SMILES file for training the substructure selector
            radius: Radius for fingerprint generation
            fp_size: Fixed size of fingerprint vector
            count: If True, use count fingerprint; otherwise, use binary
        """
        super().__init__(radius, count)
        self.radius = radius
        self.fp_size = fp_size
        self.count = count

        self.setup_fingerprint(dataset_path)

    def load_mols(self, dataset_path: Path, verbose: bool = False):
        """Load SMILES file and convert to RDKit mol objects."""
        mols = []
        failed_count = 0

        current_file_dir = Path(__file__).parent
        dataset_path = current_file_dir / dataset_path

        with open(dataset_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            smiles = line.strip()
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
                else:
                    failed_count += 1

        if verbose:
            print(f"Successfully parsed: {len(mols)} molecules")
            print(f"Failed to parse: {failed_count} SMILES")

        return mols

    def setup_fingerprint(self, dataset_path: Path, verbose: bool = False):
        """Build Sort&Slice featuriser from training molecules."""
        print("Building Sort&Slice fingerprint...")
        mols = self.load_mols(dataset_path=dataset_path, verbose=verbose)
        self.fpgen = create_sort_and_slice_ecfp_featuriser(
            mols_train=mols,
            max_radius=self.radius,
            sub_counts=self.count,
            vec_dimension=self.fp_size,
            print_train_set_info=False,
        )

    @lru_cache(maxsize=cache_size)  # Cache fingerprints for BO performance
    def __call__(self, smiles: str):
        """
        Convert SMILES string to Sort&Slice fingerprint.

        Returns RDKit UIntSparseIntVect for compatibility with Tanimoto similarity.
        """
        mol = self._validate_smiles(smiles)
        fp_array = self.fpgen(mol)

        # Convert numpy array to RDKit UIntSparseIntVect
        # Find non-zero indices and their values
        nonzero_indices = np.nonzero(fp_array)[0]

        fp_rdkit = UIntSparseIntVect(self.fp_size)
        for idx in nonzero_indices:
            fp_rdkit[int(idx)] = int(fp_array[idx])

        return fp_rdkit

    def get_fp_type(self) -> str:
        return f"sortslice{self.fp_size}-r{self.radius}"
