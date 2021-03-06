# dissertation
Code for my PhD at the University of Illinois at Urbana-Champaign
Noah W. Garfinkle
*garfink2@illinois.edu*

<p align="center">
  <a href="https://noahgarfinkle.github.io">
  </a>

  <h3 align="center">dissertation</h3>

  <p align="center">
    My dissertation
    <br>
    <a href="https://noahgarfinkle.github.io"><strong>See me on GITHUB»</strong></a>
    <br>
    <br>
    <a href="https://noahgarfinkle.github.io">Item 1</a>
    ·
    <a href="https://noahgarfinkle.github.io">Item 2</a>
    ·
    <a href="https://noahgarfinkle.github.io">Item 3</a>
  </p>
</p>

<br>

## Table of contents

- [Purpose](#purpose)
- [Status](#status)
- [Requirements](#requirements)
- [functions I need to replicate from R](functions-I-need-to-replicate-from-r)

## Purpose
> To get a PhD

## Status

## Requirements
* Python 2.7
 * Geopandas

## Installation

## LICENSE

## CHANGELOG

## Examples
```Python
    def foo():
      if not bar:
        return True
```

## 10-12 FEBRUARY Tasks
- [x] Create demo XML file
- [x] Write XML parser
- [ ] Area of Interest
- [ ] Site Suitability
  - [ ] Distance(qafRaster,df)
    - [ ] RasterizeGeodataframe
- [ ] Site Configuration
- [ ] Site Evaluation
- [ ] Site Relations
- [ ] Re-evaluated approach to first pass
  - [ ] GEOJSON AOI - exclusion cutouts + spacing = gridded dataframe of potential solutions, centerpoints of raster cells
  - [x] Filter vector dataframe with bounding box and by properties to decrease size
  - [ ] Distance from point in solution dataframe to filtered vector dataFrame
  - [x] Extract from raster within distance of point and convert to array for custom analysis/ resampling
  - [ ] Plot solution dataframe to raster with n/a where no solutions considered and ability to place any of the scores from the dataframe, saving to a geotiff

## 17 February Tasks
- [x] Get comments in all files
- [x] Clean up all code
- [ ] Get routing to write to dataframes and connect plotting to SpatialIO

## Tasks
- [x] set up a repo
- [x] convert all current functions from R to Python
- [x] Read in and manipulate rasters
- [x] Computer Euclidean distance
- [x] Compute routing distance
- [x] Set up pgrouting
- [x] test commiting using Atom

## Functions I need to replicate from R
- [ ] CRS
- [ ] spTransform
- [ ] extent
- [ ] writeRaster
- [ ] projection
- [ ] raster(matrix)
- [ ] raster(path)
- [ ] readOGR
- [ ] rasterize
- [ ] mosaic
- [ ] res
- [ ] aggregate
- [ ] resample
- [ ] distance
- [ ] weightRaster
- [ ] reclassify
- [ ] plot
- [ ] subs
- [ ] crop
