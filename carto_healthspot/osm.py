#!/usr/bin/env python3

import pathlib
import subprocess as sp
import tempfile

import geojson
import overpass


def query(countrycode: str, kind: str, **kwargs):
    """Build the query.

    Parameters
    ----------
    countrycode : str
        the country code (2 chars)
    kind : str
        health|admin
    **kwargs :
        level for admin

    Returns
    -------
    query :  str
        a query for overpass
    """
    assert len(countrycode) == 2

    op_query = [
        f'area["boundary"="administrative"]["ISO3166-1"="{countrycode.upper()}"]->.country;'
    ]
    op_verb = "body"
    if kind == "admin":
        level = kwargs.get("level", 6)
        op_query.append(f'rel["boundary"="administrative"]["admin_level"="{level}"](area.country);')
        op_query.append("out;")
        op_query.append(">;")

    elif kind == "health":
        op_query.append("(")
        op_query.append('    nwr["healthcare"](area.country);')
        op_query.append('    nwr["amenity"="clinic|doctors|hospital|pharmacy"](area.country);')
        op_query.append(");")
        op_verb = "tags center"

    elif kind == "mines":
        op_query.append("(")
        op_query.append('   nwr["landuse"="quarry"](area.country);')
        op_query.append('   nwr["man_made"="adit"](area.country);')
        op_query.append('   nwr["man_made"="mine_shaft"](area.country);')
        op_query.append(");")
        op_query.append("out;")
        op_query.append(">;")

    return "\n".join(op_query), op_verb


def request(countrycode: str, kind: str, outpath: pathlib.Path, **kwargs):
    """Request to overpass data."""

    api = overpass.API(timeout=5000, endpoint="https://overpass-api.de/api/interpreter")
    qry, verb = query(countrycode, kind, **kwargs)
    res = api.get(
        query=qry,
        verbosity=verb,
        responseformat="xml",
    )

    with tempfile.NamedTemporaryFile("w", prefix="overpass_reply_", suffix=".osm") as fout:
        fout.write(res)
        print(fout.name)

        proc = sp.run(["osmtogeojson", fout.name], capture_output=True)
        geo_data = geojson.loads(proc.stdout)

    if kind in ["admin", "mines"]:
        # filter out lines and points
        print("filtering", len(geo_data.features))
        geo_data.features = [
            feat for feat in geo_data.features if feat.geometry.type in ["Polygon", "MultiPolygon"]
        ]
        print("filtering", len(geo_data.features))

    # Save it.
    with open(outpath, "w") as gj_fout:
        geojson.dump(geo_data, gj_fout)


# request("CM", "admin", level=4)
# request("CM", "health", pathlib.Path("health.geojson"))
# request("CM", "mines", pathlib.Path("mines.geojson"))
