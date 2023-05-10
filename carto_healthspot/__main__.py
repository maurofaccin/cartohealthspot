#!/usr/bin/env python3
"""Entry point for the cli."""


import pathlib
import shutil

import click
from xdg import BaseDirectory

from carto_healthspot import utils

CACHEDIR = pathlib.Path(BaseDirectory.save_cache_path("cartohs"))


def _cache_path(kind: str = ""):
    """Filenames for temp files."""
    ctx = click.get_current_context()
    projname = ctx.parent.params["projname"]

    # make project cache folder
    local_cache = CACHEDIR / projname
    local_cache.mkdir(parents=True, exist_ok=True)

    if kind == "population":
        return local_cache / "pop.tif"
    if kind == "cases":
        return local_cache / "cases.tif"

    return local_cache


@click.group()
@click.argument("projname")
def cli(projname):
    """Compute disease distribution and hotspots.

    PROJNAME: A name for your current project (for caching purposes).
    """


# Extract Population
@cli.command()
@click.option(
    "-c", "--cutline", required=False, type=pathlib.Path, help="Geojson shape of the region."
)
@click.argument("worldpop", type=pathlib.Path)
def extract_pop(worldpop: pathlib.Path, cutline: pathlib.Path):
    """Extract population from WORLDPOP.

    Expected GeoTIFF.
    """
    output_filepath = _cache_path("population")

    if output_filepath.is_file():
        utils.log(f"File `{output_filepath}` already exists. Remove it if needed.")
        return None
    from carto_healthspot.extract_worldpop import extract_worldpop

    extract_worldpop(worldpop, output_filepath, cutline=cutline)


# disaggregate cases
@cli.command()
@click.option(
    "--healthsites",
    required=False,
    type=pathlib.Path,
    help="Location of health sites (filepath to a geojson).",
)
@click.option(
    "--mines",
    required=False,
    type=pathlib.Path,
    help="Location of mining sites (filepath to a geojson).",
)
@click.argument("zone_shapes", type=pathlib.Path)
@click.argument("output", type=pathlib.Path)
def cases(
    zone_shapes: pathlib.Path,
    healthsites: pathlib.Path | None = None,
    mines: pathlib.Path | None = None,
    output: pathlib.Path | None = None,
):
    """Disaggregate reported cases ZONE_SHAPES.

    ZONE_SHAPES should contain the shape of each health zone
    with the corresponding reported positive cases.
    Expected GeoJSON format.

    OUTPUT: GeoTIFF file to write the incidence rate.
    """
    input_filepath = _cache_path("population")

    from carto_healthspot.compute_cases import disaggregate_cases

    disaggregate_cases(
        input_filepath, zone_shapes, output, hfac_path=healthsites, mine_path=mines
    )


@cli.command()
@click.argument("OUTPUT", type=pathlib.Path)
def export_html(OUTPUT: pathlib.Path):
    """Export the incidence rate to a HTML map."""
    cases_filepath = _cache_path("cases")

    from carto_healthspot.export import export2html
    export2html(cases_filepath, OUTPUT)


@cli.command()
def urban():
    """Risk in urban areas."""
    raise NotImplementedError


@cli.command()
@click.option("-a", "--args", type=str, help="Optional args e.g.: `level=6`")
@click.argument("ISO2", type=str)
@click.argument("kind", type=str)
@click.argument("path", type=pathlib.Path)
def osm(iso2, kind, path, args):
    """Retrieve osm data."""
    from geo import osm

    kwargs = {}
    if args is not None:
        for arg in args.split():
            k, v = arg.split("=")
            kwargs[k] = v

    print(kwargs)
    osm.request(iso2, kind, path, **kwargs)


@cli.command()
def clean():
    """Clean cache files."""
    cache_dir = _cache_path()
    print("Cleaning cache folder: ", cache_dir)
    shutil.rmtree(cache_dir, onerror=lambda x, y, z: print(f"Folder '{y}' is already clean."))


if __name__ == "__main__":
    cli()
