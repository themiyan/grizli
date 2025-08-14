import os
import pytest
import numpy as np

from grizli.model import GrismFLT

# Define paths to test data.
# The user should provide these files for testing.
# Based on the user's instructions, we create a placeholder test that will be
# skipped if the data files are not found.
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
GRISM_FILE = os.path.join(TEST_DATA_PATH, 'jwXXXX_wfss.fits')
DIRECT_FILE = os.path.join(TEST_DATA_PATH, 'jwXXXX_direct.fits')
SEG_FILE = os.path.join(TEST_DATA_PATH, 'seg.fits')
REF_FILE = os.path.join(TEST_DATA_PATH, 'ref.fits')

# Condition to skip the test if data is not available
HAS_TEST_DATA = all([os.path.exists(f) for f in [GRISM_FILE, DIRECT_FILE, SEG_FILE, REF_FILE]])

@pytest.mark.skipif(not HAS_TEST_DATA, reason="Test data not found in grizli/tests/data/")
def test_parallel_correctness():
    """
    Test that the parallel implementation of compute_full_model produces
    a scientifically equivalent result to the serial implementation.
    """
    # Serial run
    flt_serial = GrismFLT(grism_file=GRISM_FILE,
                          direct_file=DIRECT_FILE,
                          seg_file=SEG_FILE,
                          ref_file=REF_FILE,
                          pad=(100, 100))

    ids = np.unique(flt_serial.seg)[1:]

    # Limit to a small number of objects for a fast CI-style test
    if len(ids) > 100:
        ids = ids[:100]

    flt_serial.compute_full_model(ids=ids, n_processes=1)
    model_serial = flt_serial.model.copy()

    # Parallel run
    flt_parallel = GrismFLT(grism_file=GRISM_FILE,
                            direct_file=DIRECT_FILE,
                            seg_file=SEG_FILE,
                            ref_file=REF_FILE,
                            pad=(100, 100))

    # Use a small number of processes for the test
    flt_parallel.compute_full_model(ids=ids, n_processes=2, chunk_size=32)
    model_parallel = flt_parallel.model.copy()

    # Compare results
    assert np.allclose(model_serial, model_parallel, rtol=0, atol=1e-6)
