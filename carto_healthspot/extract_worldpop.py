#!/usr/bin/env python3
"""Extract population count for the region of interest with the right projection."""

import pathlib

import geojson
import geopandas as geopd
import rasterio
from rasterio import mask, warp
from h3ronpy.pandas.raster import raster_to_geodataframe

from carto_healthspot import utils


def extract_worldpop(input_path: pathlib.Path, out_path: pathlib.Path, cutline=None):
    """Extract data from worldpop.

    This will reproject if necessary and cut to a cutline.
    Will save a cache file.
    """
    # set the destination crs.
    dst_crs = rasterio.CRS.from_epsg(4326)

    # read data from worldpop
    with rasterio.open(input_path, mode="r") as src:
        # transf, w, h = warp.calculate_default_transform(
        #     src.crs, dst_crs, src.width, src.height, *src.bounds
        # )
        if cutline is not None:
            # if a cutline is provided
            # compute bounds of the cutline
            # compute size in pixels
            cl_feats = geopd.read_file(cutline)
            cl_bounds = cl_feats.total_bounds

            # snap to grid
            cl_bounds, cl_shape = snap_to_grid(
                cl_bounds, [src.width, src.height], src.bounds, pad=10
            )
            utils.log(f"A cutline is provided and output will be cut at: {cl_bounds}")
            transform, w, h = warp.calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                # *bounds["size"],
                *cl_bounds,
                dst_width=cl_shape[0],
                dst_height=cl_shape[1],
            )
            # geoms = load_shapes(cutline)
            geoms = [f["geometry"] for f in cl_feats.iterfeatures()]
            out_img, out_trans = mask.mask(src, geoms, nodata=src.nodata)
        else:
            # otherwise just use the original size and bounds.
            transform, w, h = warp.calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

        # copy the rest of metadata
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": w,
                "height": h,
                "compress": "lzw",
            }
        )

        b1 = src.read(1)
        utils.log(f"Original population: {b1[b1 > 0].sum()}")
        utils.log(f"Reprojecting from {src.crs} to {dst_crs}.")

        # warp and save
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                if cutline is None:
                    src_band = src.read(i)
                else:
                    src_band = out_img[i - 1, :, :]
                utils.log(f"Population (for band {i}): {src_band[src_band > 0].sum()}")

                # TODO: avoid reprojecting if not necessary
                #       (when source and destination CRS are the same)
                warp.reproject(
                    source=src_band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    #
                    destination=rasterio.band(dst, i),
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    #
                    resampling=warp.Resampling.nearest,
                    num_threads=6,
                )


def snap_to_grid(bounds, orig_shape, orig_bounds, pad=0):
    """Snap bounds to cells of a grid."""

    def toint(num: float, b1: float, b2: float, lenght: int, t: int = 0) -> int:
        delta = (b2 - b1) / lenght
        num_cells = int((num - b1) / delta) + t
        return b1 + delta * (num_cells), num_cells

    b = [
        toint(bounds[0], orig_bounds[0], orig_bounds[2], orig_shape[0], t=-pad),
        toint(bounds[1], orig_bounds[1], orig_bounds[3], orig_shape[1], t=-pad),
        toint(bounds[2], orig_bounds[0], orig_bounds[2], orig_shape[0], t=pad + 1),
        toint(bounds[3], orig_bounds[1], orig_bounds[3], orig_shape[1], t=pad + 1),
    ]

    new_bounds = [_b[0] for _b in b]
    __check_bounds__(bounds, new_bounds)
    return new_bounds, (b[2][1] - b[0][1], b[3][1] - b[1][1])


def load_shapes(geojson_path):
    """Load shapes from path."""
    with open(geojson_path, "rt") as fin:
        feats = geojson.load(fin)

    return [f["geometry"] for f in feats.features]


def pop_to_h3(
    raster_path: pathlib.Path, h3_file_path: pathlib.Path, h3_resolution: int
):
    with rasterio.open(raster_path, mode="r") as src:
        data = src.read(1)
        profile = src.profile

    h3_hex = raster_to_geodataframe(
        data,
        profile["transform"],
        h3_resolution,
        nodata_value=profile["nodata"],
        compact=False,
        geo=True,
    )

    h3_hex.to_file(h3_file_path)


def __check_bounds__(bin, bout):
    """Check if bounds bin include bout."""
    assert bin[0] >= bout[0]
    assert bin[1] >= bout[1]
    assert bin[2] <= bout[2]
    assert bin[3] <= bout[3]
