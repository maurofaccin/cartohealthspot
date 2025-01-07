#!/usr/bin/env python3
"""Utility functions.

This script contains utility functions that are used at various stages of the process.
"""

import copy
import itertools as it
import os
import pathlib
import subprocess
from concurrent import futures
from functools import partial

import affine
import geojson
import numpy as np
import pyproj
import rasterio
import rasterio.features
import toml
from matplotlib import cm, colors, pyplot
from PIL import Image
from scipy import ndimage, signal
from shapely import geometry, ops

# matplotlib.use("Agg")
BASE = os.path.abspath("..")
try:
    with open(os.path.join(BASE, "config.toml"), "rt") as conf_file:
        CONFIG = toml.load(conf_file)
except FileNotFoundError:
    CONFIG = {}
FILTERS = {
    "military": "military=barraks landuse=military building=barraks",
    "mines": "landuse=quarry man_made=adit =mine_shaft",
    "health": "amenity=clinic =pharmacy =doctors =hospital",
    "residential": "landuse=residential",
    "building": (
        "building=yes =apartments =bungalow =cabin =detached =dormitory"
        " =house =residential =semidetached_house =mixed"
    ),
}

RISK_CMAP = copy.copy(cm.YlOrRd)
RISK_CMAP.set_bad(alpha=0.0)
RISK_CMAP.set_under(alpha=0.0)
RISK_CMAP.set_over(alpha=1.0)


def _prepate_fig_for(shape, projection=None):
    """Help function to prepare a figure to plot the geojson."""
    fig = pyplot.figure(figsize=(shape[1], shape[0]), dpi=1)
    axis = pyplot.axes([0, 0, 1, 1], projection=projection)
    axis.set_axis_off()
    if projection is not None:
        axis.outline_patch.set_linewidth(0)
    return fig, axis


def rasterize(features, bounds, shape, values=None):
    """Rasterize a shape. Return a matrix of a given shape of ones and zeros."""
    if len(features) == 0:
        return np.zeros(shape)
    transform = rasterio.transform.from_bounds(*bounds, *shape[::-1])
    if values is None:
        values = np.ones(len(features))
    inc = rasterio.features.rasterize(
        [(f.geometry, v) for f, v in zip(features, values)],
        out_shape=shape,
        fill=0.0,
        transform=transform,
    )
    return inc


def area(polygon, coords="latlong"):
    """Return the area of a given shape in square meters."""
    proj = partial(
        pyproj.transform,
        pyproj.Proj(proj=coords),
        pyproj.Proj(proj="aea", lat_1=polygon.bounds[1], lat_2=polygon.bounds[3], datum="WGS84"),
    )

    return ops.transform(proj, polygon).area


def bound_area(bounds):
    """Return area of rectangle defined by `bounds` in square meters."""
    west, south, east, north = bounds
    square = geometry.Polygon(
        [
            [west, south],
            [west, north],
            [east, north],
            [east, south],
            [west, south],
        ]
    )
    return area(square)


def snap_to_int(fnum, atol=1e-10):
    """Return the next int if close enough, otherwise the previous."""
    inum = fnum // 1 + 1
    if np.abs(fnum - inum) < atol:
        return int(inum)
    return int(inum) - 1


def lola2indx(lonlat, bbox, shape, atol=1e-10):
    """From longitude and latitude return the indices of the matrix."""
    indx = (
        snap_to_int(  # axis 0 is lat
            (bbox[3] - lonlat[1]) * shape[0] / (bbox[3] - bbox[1]), atol=atol
        ),
        snap_to_int(  # axis 1 is long
            (lonlat[0] - bbox[0]) * shape[1] / (bbox[2] - bbox[0]), atol=atol
        ),
    )
    return indx


def indx2lola(indx, bbox, shape, anchor="bottomleft"):
    """From the pixel index return longitude and latitude."""
    step = [(bbox[3] - bbox[1]) / shape[0], (bbox[2] - bbox[0]) / shape[1]]
    lonlat = [bbox[3] - indx[0] * step[0], bbox[0] + indx[1] * step[1]]
    if anchor == "center":
        lonlat[0] -= 0.5 * step[0]
        lonlat[1] += 0.5 * step[1]
    return lonlat


def point_distance(point1, point2):
    """Distance of two points in meters along the geodesic."""
    geod = pyproj.Geod(ellps="WGS84")
    _, _, distance = geod.inv(point1.x, point1.y, point2.x, point2.y)
    return distance


def load_geotif_band(geotiff, index=0):
    """Load population data."""
    with rasterio.open(geotiff, mode="r") as raster:
        bounds = list(raster.bounds)
        data = np.asarray(raster.read(index + 1))
        profile = raster.profile.copy()
    return (np.ma.masked_less_equal(data, 0.0, copy=True), np.array(bounds), profile)


def osm_to_o5m(filein, fileout, bounds):
    """Osmconvert wrapper."""
    if fileout.exists():
        # print('Cache exists')
        return fileout

    # print('Building cache')
    cmd = [
        "osmconvert",
        filein.absolute(),
        "-b=" + ",".join(map(str, bounds)),
        "--complete-ways",
        "--complete-multipolygons",
        "--complete-boundaries",
        f"-o={fileout.absolute()}",
    ]
    subprocess.run(cmd, check=True)
    return fileout


def o5m_to_geojson(filein, filters):
    """Osmfilter wrapper."""
    osmfile = filein.with_suffix(".osm")
    cmd = [
        "osmfilter",
        filein.absolute(),
        f"--keep={filters}",
        "--out-osm",
        # '--ignore-dependencies',
        "-o=" + str(osmfile.absolute()),
    ]
    subprocess.run(cmd, check=True)

    cmd = ["osmtogeojson", osmfile]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    data = geojson.loads(proc.stdout)

    os.remove(osmfile)
    return data


def get_osm_features(filters, region, bounds, shape, mask=None, workers=6):
    """Extract from osm an array encoding the fetures from `filters`."""
    cachedir = pathlib.Path("../cache/osm_features")
    cachedir.mkdir(parents=True, exist_ok=True)

    o5mcache = osm_to_o5m(
        pathlib.Path("..") / CONFIG[region]["osm"], cachedir / (region + ".o5m"), bounds
    )

    if isinstance(filters, str):
        filters = [filters]
    filters = " ".join([FILTERS[f] for f in filters])

    if shape[0] <= 1000 and shape[1] <= 1000:
        features = o5m_to_geojson(o5mcache, filters)
        matrix = rasterize(features.features, bounds, shape)
    else:
        matrix = 0
        features = None
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            runs = []
            for i, j in it.product(range(0, shape[0], 1000), range(0, shape[1], 1000)):
                lola0 = indx2lola((i, j), bounds, shape)
                lola1 = indx2lola((min(i + 1000, shape[0]), min(j + 1000, shape[1])), bounds, shape)

                bnds = [lola0[1], lola1[0], lola1[1], lola0[0]]
                subcache = o5mcache.parent / f"{o5mcache.stem}-{i}-{j}{o5mcache.suffix}"

                runs.append(
                    pool.submit(_filt_to_mat_, o5mcache, subcache, bnds, filters, bounds, shape)
                )
            for mats in futures.as_completed(runs):
                mat, feat = mats.result()
                matrix += mat
                if features is None:
                    features = feat
                else:
                    features.features += feat.features

    matrix[matrix > 1.0] = 1.0
    if mask is None:
        return matrix
    return np.ma.array(matrix, mask=mask, fill_value=0.0), features


def _filt_to_mat_(filein, fileout, bounds_filters, filters, bounds, shape):
    """Util func to rasterize osm feature."""
    osm_to_o5m(filein, fileout, bounds_filters)
    features = o5m_to_geojson(fileout, filters)
    # os.remove(fileout)
    return rasterize(features.features, bounds, shape), features


def gaussian_filter(array, sigma, truncate=3, mask=None):
    """Smoothify the matrix."""
    if np.ma.isMaskedArray(array):
        output = array.data.copy()
        output[array.mask] = 0.0
    else:
        output = array.copy()
    output = ndimage.gaussian_filter(output, sigma, truncate=truncate)

    if mask is None:
        if np.ma.isMaskedArray(array):
            mask = np.ma.getmaskarray(array)
        else:
            mask = np.asarray(output > 0.0)
    else:
        mask = np.asarray(mask, dtype=bool)

    fuzzy_mask = ndimage.gaussian_filter((~mask).astype(float), sigma, truncate=truncate)

    output[~mask] /= fuzzy_mask[~mask]
    output[mask] = 0.0
    output *= array.compressed().sum() / output.sum()
    output = np.ma.masked_where(mask, output)

    return output


def gaussian_kernel(shape=(21, 21), std=(3, 3)):
    """Return a 2D Gaussian kernel array."""
    if isinstance(shape, (int, np.int)):
        shape = (shape, shape)
    if isinstance(std, (int, np.int)):
        std = (std, std)
    gkern1 = signal.gaussian(shape[0], std=std[0]).reshape(shape[0], 1)
    gkern2 = signal.gaussian(shape[1], std=std[1]).reshape(shape[1], 1)
    gkern = np.outer(gkern1, gkern2)
    return gkern / gkern.sum()


def savez_masked(filename, array):
    """Save masked array to compressed file."""
    np.savez_compressed(filename, data=array.data, mask=array.mask)


def load_masked(filename):
    """Load masked array from compressed file."""
    loaded = np.load(filename, allow_pickle=True)
    return np.ma.array(loaded["data"], mask=loaded["mask"])


def pixel_size(bounds, shape):
    """Return the size of each pixel (on average) horizontally and vertically.

    In meters
    """
    sizex = point_distance(
        geometry.Point(bounds[0], bounds[1]), geometry.Point(bounds[2], bounds[1])
    )
    sizey = point_distance(
        geometry.Point(bounds[0], bounds[1]), geometry.Point(bounds[0], bounds[3])
    )

    size = np.array([sizey, sizex])
    return size / np.asarray(shape)


def mat2geojson(
    grid,
    bounds,
    levels,
    cmap=RISK_CMAP,
    fix_prop=None,
    dyn_prop=None,
):
    """Transform a matrix to a geojson of countour levels."""
    levels = np.sort(np.append(levels, grid.max()))
    print(levels)
    fprop = {"stroke-opacity": 0.5, "stroke-width": 1}
    if fix_prop is not None:
        fprop.update(fix_prop)

    x_indx, y_indx = np.meshgrid(
        np.linspace(bounds[0], bounds[2], grid.shape[1]),
        np.linspace(bounds[3], bounds[1], grid.shape[0]),
    )

    mplc = pyplot.contourf(
        x_indx, y_indx, grid, levels, colors=[cmap(i) for i in np.linspace(0, 1, len(levels) - 1)]
    )
    cont = contourf_to_geojson(mplc, geojson_properties=fprop)

    if dyn_prop is not None:
        for prop, feat in zip(dyn_prop, cont.features):
            feat.properties.update(prop)
            print(
                feat.properties["fill"],
                feat.properties["fill-opacity"],
            )
    return cont, levels


def notify(message="DONE"):
    """Notify."""
    subprocess.Popen(["notify-send", message])


def pop2cases(pop, cases, min_pop=100, half_value=500, max_rate=None):
    """Compute the incidence given the population density."""
    if max_rate is not None:
        raise NotImplementedError("max_rate not implemented yet")
    if pop.sum() <= 1:
        return None

    # incidence rate computed according to SIS compartmental model
    # at the dynamical equilibrium.
    dist = pop * (pop - min_pop) / (half_value + pop - 2 * min_pop)
    dist[dist < 0] = 0.0
    if dist.max() <= 1e-5:
        return None

    if isinstance(cases, np.ndarray):
        dist *= cases
        dist *= (cases * pop)[~pop.mask].sum() / dist.sum()
    else:
        dist *= cases / dist.sum()
    dist[pop.mask] = 0.0
    return dist.data


def deg2num(lat_deg, lon_deg, zoom):
    """Transform from lat long to slippy map index."""
    lat_rad = np.radians(lat_deg)
    # ntiles = 2.0 ** zoom
    ntiles = 1 << zoom  # bitwise operation
    xtile = int((lon_deg + 180.0) / 360.0 * ntiles)
    ytile = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * ntiles)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Transform from slippy map index to lat long."""
    # ntiles = 2.0 ** zoom
    ntiles = 1 << zoom  # bitwise operation
    lon_deg = xtile / ntiles * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / ntiles)))
    lat_deg = np.degrees(lat_rad)
    return (lat_deg, lon_deg)


def piecewise(array, points, vals):
    """Piecewise, linearly map values to the corresponing internal.

    Clip entries out of given range.
    """
    if isinstance(array, (float, int)):
        if array <= points[0]:
            return vals[0]
        if array >= points[-1]:
            return vals[-1]
        i = 0
        while points[i + 1] < array:
            i += 1
        return vals[i] + (array - points[i]) * (vals[i + 1] - vals[i]) / (points[i + 1] - points[i])

    new = np.empty_like(array)
    new[array < points[0]] = vals[0]
    for p_0, p_1, v_0, v_1 in zip(points[:-1], points[1:], vals[:-1], vals[1:]):
        where = np.logical_and(p_0 <= array, array < p_1)
        new[where] = v_0 + (array[where] - p_0) * (v_1 - v_0) / (p_1 - p_0)
    new[array >= points[-1]] = vals[-1]
    return new


def write_to_geofile(
    matrix, bounds, filename, cmap=RISK_CMAP, min_alpha=0.2, max_alpha=0.8, show=False
):
    """Write array to tiff or png using optionally a cmap."""
    if cmap is not None:
        ylorrd = cmap
        ylorrd.set_bad(alpha=0.0)
        ylorrd.set_under(alpha=0.0)
        ylorrd.set_over(color=ylorrd(1), alpha=max_alpha)
        img = ylorrd(matrix, bytes=True)
        img[:, :, 3] = 0.0
        img[matrix > 0, 3] = (np.clip(matrix[matrix > 0], min_alpha, max_alpha) * 255).astype(
            "uint8"
        )
    else:
        img = matrix

    try:
        count = img.shape[2]
    except IndexError:
        count = 1

    transform = rasterio.transform.from_bounds(*bounds, height=img.shape[0], width=img.shape[1])
    profile = {
        "driver": "GTiff",
        "dtype": img.dtype,
        "nodata": 0,
        "width": img.shape[1],
        "height": img.shape[0],
        "count": count,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "tiled": True,
        "compress": "lzw",
        "transform": transform,
        "interleave": "band",
    }

    ext = filename.suffix
    if ext in [".tif", ".tiff", ".geotif"]:
        with rasterio.open(filename, "w", **profile) as dst:
            if cmap is not None:
                dst.write(np.rollaxis(img, 2), indexes=[1, 2, 3, 4])
            else:
                dst.write(np.rollaxis(img[:, :, np.newaxis], 2), indexes=[1])
    elif ext == ".png":
        with open(filename.parent / (filename.stem + ".pgw"), "w") as fout:
            # write affine transform as world file
            fout.write(affine.dumpsw(profile["transform"]))
        img = Image.fromarray(img)
        img.save(filename.parent / (filename.stem + ".png"))

    if show:
        pyplot.imshow(img)
        pyplot.show()


def incidence2color(incidence, points=None, vals=None):
    """Piecewise colorize the incidence matrix."""
    if points is None:
        points = [0.001, 0.0032, 0.01]
    if vals is None:
        vals = [0.0, 0.5, 0.9]

    color = RISK_CMAP(piecewise(incidence, points, vals))
    return colors.rgb2hex(color[:3])


def contourf_to_geojson(
    contourf,
    ndigits=5,
    unit="",
    geojson_properties=None,
):
    """
    Transform matplotlib.contourf to geojson with MultiPolygons.

    From geojsoncontour.
    """
    polygon_features = []
    for coll, level in zip(contourf.collections, contourf.levels):
        polygon = get_multi_poly(coll, ndigits)
        fcolor = colors.rgb2hex(coll.get_facecolor()[0])
        properties = {
            "stroke": fcolor,
            "stroke-width": 1,
            "stroke-opacity": 1,
            "fill": fcolor,
            "fill-opacity": 0.9,
            "title": f"{level:.2f} {unit}".strip(),
        }
        if geojson_properties is not None:
            properties.update(geojson_properties)
        polygon_features.append(geojson.Feature(geometry=polygon, properties=properties))
    return geojson.FeatureCollection(polygon_features)


def get_multi_poly(path_collection, ndigits):
    """Get polygon from countour level."""
    coords = []
    for path in path_collection.get_paths():
        polygon = []
        for linestring in path.to_polygons():
            if ndigits:
                linestring = np.around(linestring, ndigits)
            polygon.append(linestring.tolist())
        coords.append(polygon)
    return geojson.MultiPolygon(coordinates=coords)


def ellipse_kernel(diam_x, diam_y):
    """Compute a kernel with ellipse shape."""
    center_y = diam_y // 2
    center_x = diam_x // 2

    y, x = np.ogrid[-center_y : diam_y - center_y, -center_x : diam_x - center_x]
    mask = (
        center_y * center_y * x * x + center_x * center_x * y * y
        <= center_y * center_y * center_x * center_x
    )

    return mask.astype(int)
