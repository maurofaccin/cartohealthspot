#!/usr/bin/env python3

import pathlib

import folium
import numpy as np
import rasterio as rio
from folium import plugins as fplugins
from matplotlib import cm, colormaps, colors


def colormap(points: list[float] = [0, 1], original_cmap=colormaps["viridis"]):
    """Produce a colormap."""
    cols = original_cmap.resampled(len(points))
    cmap = colors.LinearSegmentedColormap.from_list(
        "dist", list(zip(points, cols(np.linspace(0, 1, len(points)))))
    )
    return cmap


def geotif2image(
    infile: pathlib.Path,
):
    with rio.open(infile.expanduser(), "r") as raster:
        data = np.squeeze(raster.read())[1, :, :]
        bounds = raster.bounds

    cmap = colormap([0, 0.01, 0.03, 0.05, 1], colormaps["Reds"])
    img = cmap(data)

    print(bounds)

    img[data == 0, :] = 0
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
    map.add_child(folium.raster_layers.ImageOverlay(data, bounds, name="Incidence rate"))
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


export2html(pathlib.Path("~/.cache/cartohs/cm/cases.tif"), "pippo.html")
