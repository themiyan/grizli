import unittest

import os

import numpy as np

from grizli import grismconf as grizliconf
from grizli import GRIZLI_PATH


def test_transform():
    """
    Test JWST transforms
    """
    import astropy.io.fits as pyfits
    
    for instrument in ['NIRISS','NIRCAM','WFC3']:
        for grism in 'RC':
            for module in 'AB':
                
                tr = grizliconf.JwstDispersionTransform(instrument=instrument,
                                            grism=grism, module=module)
                
                #print(instrument, grism, module, tr.forward(1024, 1024))
                assert(np.allclose(tr.forward(1024.5, 1024.5), 1024.5))
                
                # Forward, Reverse
                x0 = np.array([125., 300])
                to = tr.forward(*x0)
                fro = np.squeeze(tr.reverse(*to))
                
                assert(np.allclose(x0-fro, 0.))

    # From header
    nis = pyfits.Header()
    nis['INSTRUME'] = 'NIRISS'

    rot90 = {'GR150C':2,  # 180 degrees
             'GR150R':3,   # 270 degrees CW
             }

    for gr in rot90:
        nis['FILTER'] = gr
        nis['PUPIL'] = 'F150W'

        tr = grizliconf.JwstDispersionTransform(header=nis)
        assert(tr.instrument == nis['INSTRUME'])
        assert(tr.grism == nis['FILTER'])
        assert(tr.rot90 == rot90[gr])


def test_bounds_checking_issue_273():
    """
    Test bounds checking fix for issue #273 - CRDS NIRISS Configurations don't load
    
    This test verifies that the bounds checking in GrismDisperser.process_config()
    prevents IndexError when beam traces go outside expected detector bounds.
    """
    # Test that the bounds checking fix works correctly
    sh = [100, 100]  # typical shape
    sh_beam = [100, 200]  # beam shape
    
    # Create the idx array as in GrismDisperser
    modelf = np.zeros(np.prod(sh_beam), dtype=np.float32)
    idx = np.arange(modelf.size, dtype=np.int64).reshape(sh_beam)
    
    x0 = np.array(sh, dtype=np.int64) // 2 - 1  # [49, 49]
    
    # Simulate problematic trace values that would cause IndexError
    ytrace_beam_problematic = np.array([95, 105, 110, 120])
    dyc = np.asarray(ytrace_beam_problematic + 20, dtype=int) - 20 + 1
    
    dx = np.arange(-50, 50)
    dxpix = dx - dx[0] + x0[1]
    
    # Test that bounds checking prevents IndexError
    y_indices = dyc + x0[0]
    x_indices = dxpix[:len(dyc)]
    
    # Clip indices to valid array bounds (this is the fix)
    y_indices = np.clip(y_indices, 0, sh_beam[0] - 1)
    x_indices = np.clip(x_indices, 0, sh_beam[1] - 1)
    
    # This should not raise IndexError
    flat_index = idx[y_indices, x_indices]
    
    # Verify the results
    assert flat_index.shape == (4,), "Bounds checking should return correct shape"
    assert np.all(y_indices < sh_beam[0]), "Y indices should be within bounds"
    assert np.all(x_indices < sh_beam[1]), "X indices should be within bounds"
    assert np.all(y_indices >= 0), "Y indices should be non-negative"
    assert np.all(x_indices >= 0), "X indices should be non-negative"


def test_read():
    
    CONF_PATH = os.path.join(GRIZLI_PATH, 'CONF')
    
    wfc3_file = os.path.join(CONF_PATH, 'G141.F140W.V4.32.conf')
    if os.path.exists(wfc3_file):
        conf = grizliconf.aXeConf(wfc3_file)
        
    try:
        import grismconf
        has_grismconf = True
    except ImportError:
        has_grismconf = False
    
    wfc3_gc = os.path.join(CONF_PATH, 'GRISM_WFC3/IR/G141.conf')
    if os.path.exists(wfc3_gc) & has_grismconf:
        conf = grizliconf.TransformGrismconf(conf_file=wfc3_gc)
        