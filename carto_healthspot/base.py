#!/usr/bin/env python3
"""Base module."""


import pathlib
import shutil
from typing import Optional as opt

import numpy as np
import rasterio
from xdg import xdg_cache_home

CACHEDIR = xdg_cache_home() / "carto_healthspot"


def get_names(
    projname: str,
    worldpop: opt[pathlib.Path] = None,
    incidence: opt[pathlib.Path] = None,
):
    """Filenames for temp files."""
    # make project cache folder
    local_cache = CACHEDIR / projname
    local_cache.mkdir(parents=True, exist_ok=True)

    return {"population": local_cache / "pop.tif"}


def clean(projname: str):
    """Remove the cache of the given project."""
    local_cache = CACHEDIR / projname
    shutil.rmtree(local_cache)


def load_geotif_band(geotiff, index=0):
    """Load population data."""
    with rasterio.open(geotiff, mode='r') as raster:
        bounds = list(raster.bounds)
        data = np.asarray(raster.read()[index])
        profile = raster.profile.copy()
    return (
        np.ma.masked_less_equal(data, 0.0, copy=True),
        np.array(bounds),
        profile
    )
