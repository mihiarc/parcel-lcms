# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial data processing pipeline that calculates land use class proportions within parcels using high-accuracy sub-pixel zonal statistics. The system processes millions of parcels against land use raster data (LCMS CONUS) with 99% better accuracy than standard methods.

## Common Development Commands

### Running the Pipeline
```bash
# Full pipeline with default settings (sub-pixel method)
uv run python -m src.main

# Test with sample data (recommended for development)
uv run python -m src.main --sample 1000

# Quick test with dry run
uv run python -m src.main --sample 100 --dry-run

# Run with specific method
uv run python -m src.main --method subpixel  # 99% accurate (default)
uv run python -m src.main --method standard  # Legacy method
```

### Testing
```bash
# Run basic pipeline test
uv run python test_pipeline.py

# Test specific components
uv run python test_fractional_pixels.py
uv run python test_acreage_validation.py
uv run python test_exact_method.py
uv run python test_real_data_comparison.py
```

### Shell Script
```bash
# Use the provided shell script for common operations
./run_pipeline.sh --sample 1000
./run_pipeline.sh --dry-run
./run_pipeline.sh --resume
```

## Architecture

### Core Processing Flow
1. **DataLoader** (`src/data_loader.py`) - Loads parcel and raster data, validates compatibility
2. **DataPreprocessor** (`src/preprocessor.py`) - Transforms CRS, validates geometries, filters parcels
3. **OptimizedZonalStatsProcessor** (`src/zonal_processor_optimized.py`) - Main processing engine supporting three methods:
   - Sub-pixel (default): 5x5 sub-pixel rasterization for fractional coverage
   - Standard: Traditional all_touched method
   - Center: Only pixel centers
4. **ChunkManager** (`src/chunk_manager.py`) - Manages chunked processing with checkpoint/resume capability
5. **ResultAggregator** (`src/result_aggregator.py`) - Combines results, generates reports and summaries

### Key Processing Methods

The pipeline's core innovation is in `zonal_processor_optimized.py` which implements:
- **Sub-pixel method**: Divides each pixel into 5x5 sub-pixels for accurate fractional coverage calculation
- **Weighted method**: For parcels < 10 acres, uses special weighting to prevent overestimation
- **Multi-threaded processing**: Uses ProcessPoolExecutor for parallel chunk processing

### Data Flow
- Input: Parcel boundaries (GeoParquet/Shapefile) + Land use raster (GeoTIFF)
- Processing: Chunks of 5000 parcels (configurable) processed in parallel
- Output: GeoParquet/CSV/GeoJSON/Shapefile with land use percentages per parcel

### Memory Management
- Spatial chunking to limit memory usage
- Checkpoint system for fault tolerance
- Optimized raster windowing to load only necessary data

## Land Use Classes Mapping

The system maps LCMS raster values to 5 land use categories:
- AGRICULTURE: values 2, 11, 12
- DEVELOPED: values 1, 8
- FOREST: values 4, 9
- OTHER: values 5, 6, 7, 13, 14, 15
- RANGELAND/PASTURE: values 3, 10

## Key Configuration

Default settings in `src/config.py`:
- CHUNK_SIZE: 5000 parcels per chunk
- N_WORKERS: CPU count - 1
- MIN_PARCEL_AREA: 900 m² (1 pixel)
- Default CRS: EPSG:5070 (CONUS Albers)

## Performance Considerations

- Sub-pixel method processes ~1,500-2,000 parcels/second
- Memory usage peaks at 8-16 GB for full dataset
- Checkpointing allows resuming failed runs
- Progress bars and detailed logging for monitoring

## Dependencies

Managed with uv (pyproject.toml):
- Core: geopandas, rasterio, shapely, pyproj
- Processing: dask, rasterstats, tqdm
- I/O: pyogrio for fast parcel loading