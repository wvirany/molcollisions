import pytest

from molcollisions.fingerprints import CompressedFP, ExactFP


def test_fp_initialization():
    """Test that we can create ExactFP and CompressedFP instances"""
    exact_fpgen = ExactFP(radius=2, count=True)
    assert exact_fpgen.radius == 2
    assert exact_fpgen.count

    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)
    assert compressed_fpgen.radius == 2
    assert compressed_fpgen.fp_size == 2048
    assert compressed_fpgen.count


def test_fp_call():
    """Test that we can call the fingerprint on a simple molecule"""
    exact_fpgen = ExactFP(radius=2, count=True)
    exact_fp = exact_fpgen("CCO")  # ethanol
    assert exact_fp is not None

    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)
    compressed_fp = compressed_fpgen("CCO")
    assert compressed_fp is not None


def test_invalid_smiles():
    """Test that, when called, the fingerprint classes correctly handle invalid SMILES strings"""
    exact_fpgen = ExactFP(radius=2, count=True)
    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)

    # Both should raise ValueError for invalid SMILES
    with pytest.raises(ValueError, match="Could not parse SMILES"):
        exact_fpgen("invalid_smiles_123")

    with pytest.raises(ValueError, match="Could not parse SMILES"):
        compressed_fpgen("invalid_smiles_123")


def test_exact_vs_compressed():
    """Test that exact vs. compressed fingerprints are actually different"""
    exact_fpgen = ExactFP(radius=2, count=True)
    compressed_fpgen = CompressedFP(radius=2, count=True, fp_size=2048)

    smiles = "c1ccccc1c1ccccc1"  # biphenyl - repeated benzene rings

    exact_fp = exact_fpgen(smiles)
    compressed_fp = compressed_fpgen(smiles)

    # Fingerprints should have different types
    assert type(exact_fp) is not type(compressed_fp)

    # ExactFP should have variable size; CompressedFP should have fixed size
    assert len(exact_fp.GetNonzeroElements()) < 2048
    assert len(compressed_fp.ToList()) == 2048

    # ExactFP should have large range for hash keys; CompressedFP keys should be within fp_size
    exact_elements = exact_fp.GetNonzeroElements()
    max_exact_key = max(exact_elements.keys())
    assert max_exact_key > 2048

    compressed_elements = compressed_fp.GetNonzeroElements()
    max_compressed_key = max(compressed_elements.keys())
    assert max_compressed_key < 2048


def test_repeated_substructures():
    """Test that a molecule with repeated substructures results in elements > 1 in count fps, but not binary"""

    smiles = "C" * 50  # repeated carbon chain

    exact_fpgen = ExactFP(radius=2, count=True)
    exact_fpgen_binary = ExactFP(radius=2, count=False)

    exact_fp = exact_fpgen(smiles)
    exact_fp_binary = exact_fpgen_binary(smiles)

    exact_elements = exact_fp.GetNonzeroElements()
    exact_elements_binary = exact_fp_binary.GetOnBits()

    # Count fp should record repeated substructures; i.e., max element should be larger
    max_exact_element = max(exact_elements.values())
    assert max_exact_element > 1

    # Total number of elements should be the same
    num_exact_elements = len(exact_elements)
    num_exact_elements_binary = len(exact_elements_binary)
    assert num_exact_elements == num_exact_elements_binary

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
    exact_fpgen = ExactFP(radius=2, count=True)
    compressed_fpgen = CompressedFP(
        radius=2, count=True, fp_size=256
    )  # small fp to force hash collisions

    smiles = (
        "C" * 50
    )  # repeated carbon chain - similar repeated substructures will likely cause hash collisions

    exact_fp = exact_fpgen(smiles)
    compressed_fp = compressed_fpgen(smiles)

    exact_elements = exact_fp.GetNonzeroElements()
    compressed_elements = compressed_fp.GetNonzeroElements()

    # Compute total number of nonzero elements for each fp - fewer elements indicates hash collisions
    num_exact_elements = len(exact_elements)
    num_compressed_elements = len(compressed_elements)
    assert num_exact_elements > num_compressed_elements

    # Total counts should be preserved
    assert sum(exact_elements.values()) == sum(compressed_elements.values())
