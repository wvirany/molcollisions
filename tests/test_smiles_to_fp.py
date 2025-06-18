import pytest
from molcollisions import fingerprints

def test_invalid_smiles_types():
    """Test that function correctly handles invalid input types."""
    with pytest.raises((ValueError, AttributeError)):  # Either error is fine
        smiles_to_fp(123)
    
    with pytest.raises((ValueError, AttributeError)):
        smiles_to_fp(None)