import pytest

from molcollisions.fingerprints import CompressedFP, SparseFP


def test_fp_initialization():
    """Test that we can create SparseFP and CompressedFP instances"""
    sparse_fpgen = SparseFP(radius=2, count=True)
    assert sparse_fpgen.radius == 2
    assert sparse_fpgen.count

    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)
    assert compressed_fpgen.radius == 2
    assert compressed_fpgen.fp_size == 2048
    assert compressed_fpgen.count


def test_fp_call():
    """Test that we can call the fingerprint on a simple molecule"""
    sparse_fpgen = SparseFP(radius=2, count=True)
    sparse_fp = sparse_fpgen("CCO")  # ethanol
    assert sparse_fp is not None

    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)
    compressed_fp = compressed_fpgen("CCO")
    assert compressed_fp is not None


def test_invalid_smiles():
    """Test that, when called, the fingerprint classes correctly handle invalid SMILES strings"""
    sparse_fpgen = SparseFP(radius=2, count=True)
    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)

    # Both should raise ValueError for invalid SMILES
    with pytest.raises(ValueError, match="Could not parse SMILES"):
        sparse_fpgen("invalid_smiles_123")

    with pytest.raises(ValueError, match="Could not parse SMILES"):
        compressed_fpgen("invalid_smiles_123")


def test_sparse_vs_compressed():
    """Test that sparse vs. compressed fingerprints are actually different"""
    sparse_fpgen = SparseFP(radius=2, count=True)
    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)

    smiles = "c1ccccc1c1ccccc1"  # biphenyl - repeated benzene rings

    sparse_fp = sparse_fpgen(smiles)
    compressed_fp = compressed_fpgen(smiles)

    # Fingerprints should have different types
    assert type(sparse_fp) is not type(compressed_fp)

    # SparseFP should have variable size; CompressedFP should have fixed size
    assert len(sparse_fp.GetNonzeroElements()) < 2048
    assert len(compressed_fp.ToList()) == 2048

    # SparseFP should have large range for hash keys; CompressedFP keys should be within fp_size
    sparse_elements = sparse_fp.GetNonzeroElements()
    max_sparse_key = max(sparse_elements.keys())
    assert max_sparse_key > 2048

    compressed_elements = compressed_fp.GetNonzeroElements()
    max_compressed_key = max(compressed_elements.keys())
    assert max_compressed_key < 2048


def test_repeated_substructures():
    """Test that a molecule with repeated substructures results in elements > 1 in count fps, but not binary"""

    smiles = "C" * 50  # repeated carbon chain

    sparse_fpgen = SparseFP(radius=2, count=True)
    sparse_fpgen_binary = SparseFP(radius=2, count=False)

    sparse_fp = sparse_fpgen(smiles)
    sparse_fp_binary = sparse_fpgen_binary(smiles)

    sparse_elements = sparse_fp.GetNonzeroElements()
    sparse_elements_binary = sparse_fp_binary.GetOnBits()

    # Count fp should record repeated substructures; i.e., max element should be larger
    max_sparse_element = max(sparse_elements.values())
    assert max_sparse_element > 1

    # Total number of elements should be the same
    num_sparse_elements = len(sparse_elements)
    num_sparse_elements_binary = len(sparse_elements_binary)
    assert num_sparse_elements == num_sparse_elements_binary

    # Repeat same test for compressed fp:
    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)
    compressed_fpgen_binary = CompressedFP(radius=2, count=False, fp_size=2048)

    compressed_fp = compressed_fpgen(smiles)
    compressed_fp_binary = compressed_fpgen_binary(smiles)

    compressed_elements = compressed_fp.GetNonzeroElements()
    compressed_elements_binary = compressed_fp_binary.GetOnBits()

    # Count FP should record repeated substructures; i.e., max element should be larger
    max_compressed_element = max(compressed_elements.values())
    assert max_compressed_element > 1

    # Total number of elements should be the same
    num_compressed_elements = len(compressed_elements)
    num_compressed_elements_binary = len(compressed_elements_binary)
    assert num_compressed_elements == num_compressed_elements_binary


def test_hash_collisions():
    """Test that a complex molecule does result in hash collisions in a compressed FP"""
    sparse_fpgen = SparseFP(radius=2, count=True)
    compressed_fpgen = CompressedFP(
        radius=2, count=True, fp_size=256
    )  # small fp to force hash collisions

    smiles = (
        "C" * 50
    )  # repeated carbon chain - similar repeated substructures will likely cause hash collisions

    sparse_fp = sparse_fpgen(smiles)
    compressed_fp = compressed_fpgen(smiles)

    sparse_elements = sparse_fp.GetNonzeroElements()
    compressed_elements = compressed_fp.GetNonzeroElements()

    # Compute total number of nonzero elements for each fp - fewer elements indicates hash collisions
    num_sparse_elements = len(sparse_elements)
    num_compressed_elements = len(compressed_elements)
    assert num_sparse_elements > num_compressed_elements

    # Total counts should be preserved
    assert sum(sparse_elements.values()) == sum(compressed_elements.values())
