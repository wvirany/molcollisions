from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from functools import lru_cache, partial

@lru_cache(maxsize=300_000) # Cache fingerprints for faster Bayesian optimization
def smiles_to_fp(smiles: str, sparse: bool = True, radius: int = 2, fp_size: int = 2048):
    """
    Convert SMILES to ECFP fingerprint.
    
    Args:
        smiles: SMILES string representing the molecule
        sparse: If True, return sparse fingerprint; if False, return dense fingerprint  
        radius: Radius for Morgan fingerprint generation
        fp_size: Size of fingerprint vector (only used when sparse=False)
        
    Returns:
        Sparse fingerprint (if sparse=True) or dense numpy array (if sparse=False)
        
    Raises:
        ValueError: If SMILES string cannot be parsed by RDKit
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    return fpgen.GetSparseCountFingerprint(mol) if sparse else fpgen.GetCountFingerprint(mol)


def config_fp_func(sparse: bool = True, radius: int = 2, fp_size: int = 2048):
    """Create a partially configured fingerprint function with fixed parameters."""
    fp_func = partial(smiles_to_ecfp, sparse=sparse, radius=radius, fp_size=fp_size)
    return fp_func