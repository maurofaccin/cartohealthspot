#!/usr/bin/env python3
"""Utility functions."""

import logging
import pathlib
from functools import partial

import numpy as np
import pyproj
import rasterio
from shapely import geometry, ops

logging.basicConfig(level=20)


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple:
    """Transform from lat long to slippy map index.

    Parameters
    ----------
    lat_deg : float
        latitude in degrees
    lon_deg : float
        longitude in degrees
    zoom : int
        slippy map zoom level

    Returns
    -------
    index: (int, int)
        slippy map tile indeces
    """
    lat_rad = np.radians(lat_deg)
    ntiles = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * ntiles)
    ytile = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * ntiles)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple:
    """Transform from slippy map index to lat long.

    Parameters
    ----------
    xtile : int
        x index of slippy map tile
    ytile : int
        y index of slippy map tile
    zoom : int
        slippy map zoom level

    Returns
    -------
    latlon : tuple
        latitude and longitude in degrees
    """
    ntiles = 2.0**zoom
    lon_deg = xtile / ntiles * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / ntiles)))
    lat_deg = np.degrees(lat_rad)
    return (lat_deg, lon_deg)


def log(text, level='info'):
    """Log."""
    if level == "warn":
        logging.warning(text)
    else:
        logging.info(text)


def load_geotif_band(geotiff: str | pathlib.Path, index: int = 0):
    """Load data matrix from geotif band.

    Parameters
    ----------
    geotiff : str | pathlib.Path
        path to geotiff raster
    index: int
        the index of the band to extract

    Returns
    -------
    data : np.ma.masked_array
        the data (masked)
    bounds : np.ndarray
        west, south, east, north
    profile : dict
        raster profile
    """
    with rasterio.open(geotiff, mode="r") as raster:
        bounds = list(raster.bounds)
        data = np.asarray(raster.read(index + 1))
        profile = raster.profile.copy()
    return (np.ma.masked_less_equal(data, 0.0, copy=True), np.array(bounds), profile)


def box_area(west, south, east, north):
    """Transform to coordinates in meters and compute the area."""
    transformer = pyproj.Transformer.from_crs(4326, 3857)
    box = transformer.transform_bounds(west, south, east, north)
    return geometry.box(*box).area
