#!/usr/bin/env python3
"""Compute the distribution of cases.

This script use the population estimation and the local TB  reports
to estimate the local incidence rate.

The parameters of the model are defined at the beginning:

    THR1: minimum number of population per km2 to have nonzero incidence
    THR2: value of the population per kn2 at which the incidence rate reaches
        half of the maximum value in the health zone
    HFAR: multiplicative factor for locations far from facilities
    HFAC: multiplicative factor for locations close to facilities
    HHOS: multiplicative factor for locations close to hospitals

Output:
    cache/cases_dist/REGION/cases.geotif
    cache/cases_dist/REGION/hzones.geojson
    cache/cases_dist/REGION/mines.geojson
    cache/cases_dist/REGION/hfac.geojson
"""

import pathlib
from dataclasses import dataclass, field
from typing import Literal

import affine
import geopandas as geopd
import numpy as np
import pyproj
import rasterio
import shapely
from rasterio.features import rasterize as riorasterize
from scipy import ndimage, optimize
from tqdm import tqdm

from carto_healthspot import utils

THR1 = 100
THR2 = 500

HFAR = 3.0
HFAC = 2.0
HHOS = 1.0


@dataclass
class Metadata:
    """Config data to use around."""

    popfile: pathlib.Path
    profile: dict = field(init=False)
    bounds: np.ndarray = field(init=False)
    shape: tuple = field(init=False)

    def __post_init__(self):
        with rasterio.open(self.popfile, "r") as raster:
            self.bounds = np.array(raster.bounds)
            self.profile = raster.profile.copy()
            self.shape = (self.profile["height"], self.profile["width"])


def disaggregate_cases(
    popfile: pathlib.Path,
    zone_shapes: pathlib.Path,
    casesfile: pathlib.Path,
    hfac_path: pathlib.Path | None = None,
    mine_path: pathlib.Path | None = None,
    mine_incidence_rate: float = 0.005,
):
    """Compute the incidence rate from population and disease reports.

    Parameters
    ----------
    popfile : pathlib.Path
        the location of the population count file GeoTIFF.
    zone_shapes : pathlib.Path
        shapes of the health case zones reporting cases.
        number of cases should be found in feat['properties']['cases']
    casesfile :  pathlib.Path
        incidence output file (GeoTIFF)
    hfac_path : pathlib.Path
        location of health facilities (GeoJSON)
    mine_path : pathlib.Path
        location of mining activities (GeoJSON)
    """
    metadata = Metadata(popfile)
    cases = disaggregate_zones(metadata, zone_shapes, casesfile)

    utils.log(f"Total reported cases: {cases.sum()}")

    if mine_path is not None:
        utils.log("--- --- mines")
        mines = mining_incidence(
            cases, metadata, mine_path, incidence_rate=mine_incidence_rate
        )
        cases = np.maximum(cases, mines)
        utils.log(f"Total reported cases: {cases.sum()}")

    if hfac_path is not None:
        utils.log("--- --- health facilities closeness")
        hmap = health_facilities(metadata, hfac_path)
        cases *= hmap
        utils.log(f"Total reported cases: {cases.sum()}")

    with rasterio.open(popfile, "r") as inraster:
        pop = inraster.read(1, masked=True).filled(0)
        write_incidence(casesfile, cases / pop, metadata.profile)


def mining_incidence(
    cases: np.ndarray,
    metadata: Metadata,
    mine_path: pathlib.Path,
    radius: float = 2000.0,
    incidence_rate: float = 0.01,
):
    """Compute the incidence close to mining sites.

    Parameters
    ----------
    cases: np.ndarray
        already computed number of cases
    metadata : Metadata
    mine_path : pathlib.Path
        path to the mines file
    radius : int
        influence radius around mines
        (Default value = 5000 meters)
    incidence_rate : float
        Incidence rate close to mines
        (Default value = 0.01 (1%))

    Returns
    -------
    multiplier : np.ndarray
        incidence of disease around the mining exploitations
    """
    mine_feats = geopd.read_file(mine_path)

    with rasterio.open(metadata.popfile, "r") as raster:
        bounds = raster.bounds
        pop = np.asarray(raster.read(1, masked=True).filled(0))
        # mine_incidence = np.zeros((raster.profile["height"], raster.profile["width"]))

        # rasterized mines
    utils.log("rasterize")
    mines = dilated_and_blurred(metadata, mine_feats, radius, kind="approx")

    inc = (cases / pop) * (mines * 9 + 1)

    return np.clip(inc, 0, incidence_rate) * pop

    return mines * 9 + 1
    unit_area = utils.box_area(*bounds) / pop.size / 1e6  # in km^2
    inc = _effective_population(
        pop,
        min_pop=THR1 * unit_area,
        half_value=THR2 * unit_area,
    )
    return mines * incidence_rate * inc * pop
    # return mines * incidence_rate * pop


def pop2cases(
    pop: np.ma.masked_array,
    cases: float,
    min_pop: float = 100,
    half_value: float = 500,
    max_rate: float = 1.0,
):
    """Compute the incidence given the population density."""
    if pop.sum() <= 1:
        return 0.0

    # incidence rate computed according to SIS compartmental model
    # at the dynamical equilibrium.
    if True:
        kind = "r0"

        def __comp_cases__(p1, p0, pop, ncases, kind: str = "old"):
            return np.abs(
                (
                    _effective_population(pop, min_pop=p0, half_value=p1, kind=kind)
                    * pop
                ).sum()
                - ncases
            )

        opt = optimize.minimize_scalar(
            __comp_cases__,
            args=(min_pop, pop, cases, kind),
            # bounds=(min_pop, 100000 * min_pop),
            # bounds=(1, 1000000),
            # bounds=(1 / pop.max(), 100),
        )
        case_dist = (
            _effective_population(pop, min_pop=min_pop, half_value=opt.x, kind=kind)
            * pop
        )
        print(f"{cases} => {case_dist.sum()}     args: {opt.x}   {min_pop}")
        assert case_dist.sum() > cases / 2
        case_dist = np.ma.masked_less_equal(case_dist, 0)
    else:
        case_dist = (
            _effective_population(
                pop,
                min_pop=min_pop,
                half_value=half_value,
            )
            * pop
        )

        # the population is too sparse
        if case_dist.max() <= 1e-5:
            return 0.0

        # should sum up to the expected number of cases,
        # preserving the proportions
        case_dist *= cases / case_dist.sum()

    case_dist[pop.mask] = 0.0

    incidence_rate = case_dist / pop
    if incidence_rate.max() > max_rate:
        utils.log(
            f"Incidence above threshold ({incidence_rate.max():6.5f} / {max_rate}). Clipping."
        )
        case_dist = np.clip(incidence_rate, 0, max_rate) * pop
        # re-adjust
        case_dist *= cases / case_dist.sum()
    return case_dist.data


def _effective_population(
    pop: np.ma.masked_array,
    min_pop: float = 100,
    half_value: float = 500,
    kind: str = "old",
):
    if kind == "old":
        incidence = (pop - min_pop) / (half_value + pop - 2 * min_pop)
    elif kind == "r0":
        incidence = (pop - 1) / (pop + half_value)
    elif kind == "linear":
        incidence = 1 - 1 / (half_value * pop)

    return np.clip(incidence, 0.0, 1.0)


def gaussian_filter(mat, sigmas, use="ndimage"):
    """Return a blurred matrix."""
    if use == "ndimage":
        return ndimage.gaussian_filter(mat, sigmas)
    if use == "fft":
        fft_mat = np.fft.fft2(mat)
        trans = ndimage.fourier_gaussian(fft_mat, sigmas)
        return np.fft.ifft2(trans).astype("float")


def write_incidence(filepath: pathlib.Path, data: np.ndarray, profile: dict):
    """Write data to a GeoTIFF file.

    Parameters
    ----------
    filepath : pathlib.Path
        Path of the new file (will be overwritten)
    data : np.ndarray
        Data
    profile : dict
        Profile of the GeoTIFF.
    """
    profile["count"] = 1
    with rasterio.open(filepath, "w", **profile) as outraster:
        outraster.write(data, 1)
        # outraster.write(cases, 1)
    utils.log("TODO: classify!")


def rasterize(features, transform, shape, values=None):
    """Rasterize a shape. Return a matrix of a given shape of ones and zeros."""
    if len(features) == 0:
        return np.zeros(shape)
    if values is None:
        values = np.ones(len(features))
    # inc = rasterio.features.rasterize(
    inc = riorasterize(
        [(f, v) for f, v in zip(features, values)],
        out_shape=shape,
        fill=0.0,
        transform=transform,
    )
    return inc


def disaggregate_zones(
    metadata: Metadata,
    zone_shapes: pathlib.Path,
    casesfile: pathlib.Path,
):
    """Compute the incidence rate from population and disease reports.

    Parameters
    ----------
    popfile : pathlib.Path
        the location of the population count file GeoTIFF.
    zone_shapes : pathlib.Path
        shapes of the health case zones reporting cases.
        number of cases should be found in feat['properties']['cases']
    casesfile :  pathlib.Path
        incidence output file (GeoTIFF)
    """
    healthzones = geopd.read_file(zone_shapes)
    cases = 0

    with rasterio.open(metadata.popfile, "r") as raster:
        pop = np.asarray(raster.read(1))

        pop = np.ma.masked_less_equal(pop, 0.0)
        unit_area = utils.box_area(*metadata.bounds) / pop.size / 1e6  # in km^2

        for hz in tqdm(healthzones.iterfeatures(), total=len(healthzones)):
            hz_mask = rasterize(
                [hz["geometry"]], metadata.profile["transform"], metadata.shape
            )
            hz_cases = hz["properties"]["cases"]

            loc_case_distribution = pop2cases(
                pop * hz_mask,
                hz_cases,
                min_pop=THR1 * unit_area,
                half_value=THR2 * unit_area,
                max_rate=0.02,
            )
            # we do not expect any place with incidence too high
            # maximum accepted incidence is 2%
            cases += loc_case_distribution

    return cases


def geodesic_distance(p1, p2):
    """Compute the geodesic distance between two points."""
    geod = pyproj.Geod(ellps="WGS84")
    return geod.line_length([p1[0], p2[0]], [p1[1], p2[1]])


def pixel_size(transform: affine.Affine) -> tuple[float]:
    """Compute the pixel size in meters of a raster given the affine transform."""
    origin = (transform.xoff, transform.yoff)
    vx = (transform.xoff + transform[0], transform.yoff)
    vy = (transform.xoff, transform.yoff + transform[4])
    return (geodesic_distance(origin, vx), geodesic_distance(origin, vy))


def prepare_gaussian(sigma):
    """Prepare a gaussian.

    Parameters
    ----------
    sigma : tuple[int]
        sigmas along each axis in the form (sigma_x, sigma_y)

    Returns
    -------
    gauss : np.ndarray
        a gaussian distribution centered on +- 3 sigma
    """
    new_grid = np.zeros((6 * sigma[1] + 1, 6 * sigma[0] + 1))
    new_grid[3 * sigma[1], 3 * sigma[0]] = 2 * np.pi * (sigma[1] * sigma[0])
    return gaussian_filter(new_grid, sigma)


def _vector_slice_overlap(indx1: int, len1: int, indx2: int, len2: int) -> slice:
    """Return the indices of overlapping segment of two vectors where indx1 and indx2 coincide."""
    indexes1 = [0, len1]
    indexes2 = [0, len2]

    if indx1 > indx2:
        indexes1[0] = indx1 - indx2
    else:
        indexes2[0] = indx2 - indx1

    if len1 - indx1 > len2 - indx2:
        indexes1[1] = indx1 + len2 - indx2
    else:
        indexes2[1] = indx2 + len1 - indx1

    return slice(*indexes1), slice(*indexes2)


def _matrix_slice_overlap(indx1, shape1, indx2, shape2) -> tuple[slice]:
    """Return the indices of overlapping area in mat1 and mat2 where indx1 and indx2 coincide."""
    ind1x, ind2x = _vector_slice_overlap(indx1[1], shape1[1], indx2[1], shape2[1])
    ind1y, ind2y = _vector_slice_overlap(indx1[0], shape1[0], indx2[0], shape2[0])
    return ind1x, ind1y, ind2x, ind2y


def health_facilities(
    metadata: Metadata,
    hfac_path: pathlib.Path = None,
):
    """Compute a multiplicative factor near/far the health facitilies.

    It will set the following multiplicative factors:
        - close an hospital

    Parameters
    ----------
    metadata : Metadata
        User data
    hfac_path : pathlib.Path
        path to the health facility file

    Returns
    -------
    multiplier : np.ndarray
        multiplier
    """
    hfacs = geopd.read_file(hfac_path)

    hfac_hospitals = hfacs[
        (hfacs.amenity == "hospital") | (hfacs.healthcare == "hospital")
    ]
    hfac_others = hfacs[
        (hfacs.amenity != "hospital") & (hfacs.healthcare != "hospital")
    ]

    f1 = dilated_and_blurred(metadata, hfac_hospitals, 2 * 5000, kind="approx")
    f2 = dilated_and_blurred(metadata, hfac_others, 5000, kind="approx")

    return HFAR - np.maximum((HFAR - HHOS) * f1, (HFAR - HFAC) * f2)


def dilated_and_blurred(
    metadata: Metadata,
    features: geopd.GeoDataFrame,
    radius: float,
    kind: Literal["buffer", "approx"] = "buffer",
):
    """Compute a blurred dilated rasterized version of features."""
    resolution = np.array(pixel_size(metadata.profile["transform"]))
    sigma = (radius / resolution).astype(int)

    if kind == "buffer":
        buf = lenght(radius, metadata.bounds)
        utils.log(f"Using a buffer of {buf}")
        dilated = geopd.GeoSeries(
            [
                feat["geometry"].buffer((buf[0] + buf[1]) / 2)
                for _, feat in features.iterrows()
            ]
        )
        dilated_polygon = dilated.unary_union
        dilated_raster = rasterize(
            [dilated_polygon], metadata.profile["transform"], metadata.shape
        )

        # sigma needs to be that small!
        dilated_raster = np.clip(gaussian_filter(dilated_raster, sigma / 2), 0, 1.0)
    elif kind == "approx":
        gaussian = prepare_gaussian(sigma)
        gaussian_center = ((gaussian.shape[0] - 1) // 2, (gaussian.shape[1] // 2))

        dilated_raster = np.zeros(metadata.shape)
        for indx, feat in tqdm(features.iterrows(), total=len(features)):
            for point in relevant_points(feat["geometry"]):
                center_indx = list(
                    map(int, metadata.profile["transform"].__invert__() * point)
                )[::-1]
                ov_ind = _matrix_slice_overlap(
                    center_indx, dilated_raster.shape, gaussian_center, gaussian.shape
                )
                dilated_raster[ov_ind[1], ov_ind[0]] = np.maximum(
                    dilated_raster[ov_ind[1], ov_ind[0]], gaussian[ov_ind[3], ov_ind[2]]
                )

    return dilated_raster


def relevant_points(feature: shapely.geometry.base.BaseGeometry):
    """Find a handful of relevant points."""
    if feature.geom_type == "Point":
        return feature.coords[:]
    else:
        simplified = feature.simplify(0.1, preserve_topology=True)
        return simplified.exterior.coords


def lenght(radius: float, bounds: np.ndarray):
    """Return the lenght of a segment of `radius` meters in the current CRS."""
    dx = geodesic_distance((bounds[0], bounds[1]), (bounds[2], bounds[1]))
    dy = geodesic_distance((bounds[0], bounds[1]), (bounds[0], bounds[3]))

    lx = (bounds[2] - bounds[0]) * radius / dx
    ly = (bounds[3] - bounds[1]) * radius / dy

    return lx, ly
