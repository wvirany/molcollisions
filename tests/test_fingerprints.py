from molcollisions.fingerprints import CompressedFP, SparseFP


def test_fp_creation():
    """Test that we can create SparseFP and CompressedFP instance."""
    sparse_fp = SparseFP(radius=2, count=True)
    assert sparse_fp.radius == 2
    assert sparse_fp.count

    compressed_fp = CompressedFP(radius=2, count=True, fp_size=2048)
    assert compressed_fp.radius == 2
    assert compressed_fp.count
    assert compressed_fp.fp_size == 2048


def test_fp_call():
    """Test that we can call the fingerprint on a simple molecule."""
    sparse_fp = SparseFP(radius=2, count=True)
    result = sparse_fp("CCO")  # ethanol
    assert result is not None

    compressed_fp = CompressedFP(radius=2, count=True, fp_size=2048)
    result = compressed_fp("CCO")
    assert result is not None
