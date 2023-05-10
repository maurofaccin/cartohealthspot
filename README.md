# CARTOTB

This software is will load information about:

- population density ([Worldpop](https://www.worldpop.org/))
- OSM shapes (use overpass API)
- mines from OSM or other
- TB incidence

And provides a risk map.

## Installation

Install as:

```
pip install --user .
```

## Usage

### Data needed

#### Population

From [Worldpop](https://www.worldpop.org/) download `population density`.
Select "Unconstrained individual countries UN adjusted".

TODO: try "Constrained" version.

```
xxx_ppp_2020_UNadj.tif
```

#### Shapes and incidence

A feature collection of (multi)polygons covering the region of interest.
Mandatory feature properties are:

- `name`: name of the feature
- `num_cases`: number of (TB) cases

```
xxx_shapes.geojson
```

#### Mines (Optional)

Feature collection of Points and (Multi)polygons that describe mines and their locations.

```
xxx_mines.geojson
```

#### Health facilities locations (Optional)

Feature collection of Points and (Multi)polygons that describe mines and their locations.

Feature properties:

- `name`: health facility name
- `type`: specialized/hospital/other

```
xxx_health.geojson
```

#### Other

Other, non implemented yet, may be:

- feedback of ACF actions (Location of positive and negative cases)
