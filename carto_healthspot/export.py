#!/usr/bin/env python3

import pathlib

import folium
import numpy as np
import rasterio as rio
from matplotlib import colormaps, colors


def colormap(
    points: int = 2, original_cmap: colors.Colormap = colormaps["viridis"]
) -> colors.Colormap:
    """Produce a colormap."""
    if isinstance(original_cmap, str):
        original_cmap = colormaps[original_cmap]
    cols = original_cmap.resampled(points)

    # get the segmented
    segs = np.linspace(0, 1, points)
    segs = list(zip(segs, cols(segs)))
    segs[0][1][-1] = 0.0
    cmap = colors.LinearSegmentedColormap.from_list("dist", segs).resampled(256)
    return cmap


def geotif2image(
    infile: pathlib.Path,
    vmin: float = 0.0001,
    vmax: float = 0.01,
    return_cmap: bool = False,
    shape: tuple[int] | None = None,
):
    with rio.open(infile.expanduser(), "r") as raster:
        data = np.squeeze(raster.read(out_shape=shape))
        if data.ndim == 3:
            data = data[1, :, :]
        bounds = raster.bounds
        print(raster.crs)

    cmap = colormap(points=5, original_cmap="Reds")
    img = cmap(np.clip(np.log10(data / vmin) / np.log10(vmax / vmin), 0, 1))

    if return_cmap:
        return img, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], cmap
    return img, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


def export2html(infile: pathlib.Path, outfile: pathlib.Path):
    """Save an HTML map."""
    data, bounds = geotif2image(infile)
    print(bounds)

    map = folium.Map(
        location=[(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2],
        zoom_start=7,
        # tiles="cartodbdark_matter",
        tiles="Stamen Terrain",
        name="Background",
    )
    map.add_child(
        folium.raster_layers.ImageOverlay(data, bounds, name="Incidence rate")
    )
    map.add_child(
        folium.features.Choropleth(
            "/home/mauro/curro/projs/2020_drug_resistance_Cameroon/data_hs/gadm41_CMR_1.geojson",
            fill_opacity=0,
            name="Regions",
        )
    )
    map.add_child(folium.LayerControl())
    map.save("test.html")

    print(data.shape)
