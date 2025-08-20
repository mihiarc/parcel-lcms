# Parcel Land Use Zonal Statistics Pipeline

Calculate land use class proportions within each parcel using zonal statistics.

## Features

- **Memory-efficient processing**: Handles millions of parcels against large rasters
- **CRS transformation**: Automatic alignment of different coordinate systems  
- **Chunked processing**: Spatial and count-based chunking strategies
- **Checkpoint/Resume**: Fault-tolerant processing with resume capability
- **Multiple output formats**: GeoParquet, GeoJSON, Shapefile, CSV
- **Comprehensive validation**: Geometry validation and proportion checks

## Installation

```bash
# Install dependencies with uv
uv sync
```

## Usage

### Basic Usage

```bash
# Process all parcels
uv run python -m src.main

# Process with custom chunk size
uv run python -m src.main --chunk-size 10000

# Test with sample data
uv run python -m src.main --sample 1000
```

### Command Line Options

- `--raster PATH`: Path to land use raster (default: data/LCMS_CONUS_v2024-10_Land_Use_2024.tif)
- `--parcels PATH`: Path to parcel boundaries (default: data/ParcelsWithAssessments.parquet)
- `--chunk-size N`: Number of parcels per chunk (default: 5000)
- `--strategy {count,spatial,hybrid}`: Chunking strategy (default: hybrid)
- `--sample N`: Process only N parcels for testing
- `--resume`: Resume from checkpoint
- `--dry-run`: Analyze data without processing
- `--output-format {geoparquet,geojson,shapefile,csv}`: Output format
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging verbosity

### Python API

```python
from src.main import run_pipeline

# Run with custom parameters
run_pipeline(
    sample_size=1000,
    chunk_size=5000,
    strategy="hybrid",
    output_format="geoparquet"
)
```

## Land Use Classes

1. **AGRICULTURE**: Croplands, orchards, vineyards, livestock operations
2. **DEVELOPED**: Residential, commercial, industrial, transportation
3. **FOREST**: Natural forest, plantations, woody wetlands (>10% tree cover)
4. **OTHER**: Water bodies, snow/ice, salt flats
5. **RANGELAND/PASTURE**: Native grasses, shrubs, managed pastures

## Output Fields

- `PARCEL_LID`: Unique parcel identifier
- `CAL_ACREAGE`: Calculated parcel acreage
- `agriculture_pct`: Percentage of parcel in agriculture
- `developed_pct`: Percentage of parcel in developed land
- `forest_pct`: Percentage of parcel in forest
- `other_pct`: Percentage of parcel in other land uses
- `rangeland_pasture_pct`: Percentage of parcel in rangeland/pasture
- `majority_land_use`: Dominant land use class
- `total_pixels`: Total pixel count
- `valid_pixels`: Valid (non-NoData) pixel count
- `geometry`: Parcel geometry

## Performance

- **Memory usage**: 8-16 GB peak
- **Processing speed**: ~100-500 parcels/second
- **Estimated time**: 3-6 hours for 2 million parcels

## Troubleshooting

If you encounter GDAL warnings, they can be safely ignored. The pipeline will continue to function correctly.