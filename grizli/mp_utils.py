# grizli/mp_utils.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

def _compute_partial_model(args):
    """
    Worker: compute partial contamination model for a chunk of object IDs.

    Parameters
    ----------
    args : tuple
        (flt_ctor_kwargs, ids, mags, size, min_size, compute_size, store)

    Returns
    -------
    np.ndarray  or  (np.ndarray, dict)
        Summed partial model; optionally per-object beams if store=True.
    """
    from .model import GrismFLT  # import here to keep pickling clean

    (flt_ctor_kwargs, ids, mags,
     size, min_size, compute_size, store) = args

    flt = GrismFLT(**flt_ctor_kwargs)

    partial = np.zeros_like(flt.model, dtype=np.float32)
    object_disp = {} if store else None

    # Ensure we don't mutate the flt.model inside workers
    for oid, mag in zip(ids, mags):
        out = flt.compute_model_orders(
            id=oid,
            mag=mag,
            size=size,
            min_size=min_size,
            compute_size=compute_size,
            in_place=False,   # critical: produce isolated array
            store=store
        )
        if out is False:
            continue
        beams, obj_model = out
        partial += obj_model.astype(np.float32, copy=False)
        if store:
            object_disp[oid] = beams

    return partial if not store else (partial, object_disp)
