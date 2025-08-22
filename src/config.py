"""Configuration settings for the zonal statistics pipeline."""
import os
from pathlib import Path
import logging
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Input data paths
RASTER_PATH = DATA_DIR / "LCMS_CONUS_v2024-10_Land_Use_2024.tif"
PARCEL_PATH = DATA_DIR / "ParcelsWithAssessments.parquet"

# Parcel data field names
PARCEL_ID_FIELD = "PARCEL_LID"  # Unique parcel identifier
PARCEL_ACREAGE_FIELD = "CAL_ACREAGE"  # Calculated acreage field

# CRS definitions
RASTER_CRS = "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
PARCEL_CRS = "EPSG:4326"  # WGS84
OUTPUT_CRS = RASTER_CRS  # Use raster CRS for output

# Land use class definitions
LAND_USE_CLASSES = {
    1: "AGRICULTURE",
    2: "DEVELOPED",
    3: "FOREST",
    4: "OTHER",
    5: "RANGELAND_PASTURE"
}

# Detailed land use descriptions
LAND_USE_DESCRIPTIONS = {
    1: "Land used to produce food, fiber, and fuels (croplands, orchards, vineyards, livestock operations)",
    2: "Land covered by man-made structures (residential, commercial, industrial, transportation)",
    3: "Land with 10% or greater tree cover (natural forest, plantations, woody wetlands)",
    4: "Land covered with snow/ice, water, salt flats (permanent water bodies, glaciers)",
    5: "Native grasses, shrubs, forbs (rangeland) or managed grass species (pasture)"
}

# Processing parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))  # Parcels per chunk
RASTER_WINDOW_SIZE = int(os.getenv("RASTER_WINDOW_SIZE", "10000"))  # Pixels
MAX_MEMORY_GB = float(os.getenv("MAX_MEMORY_GB", "16"))
N_WORKERS = int(os.getenv("N_WORKERS", "12"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "100"))  # Chunks

# Parallel processing parameters
PARALLEL_PROCESSING = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
MAX_PARALLEL_CHUNKS = int(os.getenv("MAX_PARALLEL_CHUNKS", "10"))  # Max chunks in flight
WORKER_MEMORY_LIMIT = int(os.getenv("WORKER_MEMORY_LIMIT", "2048"))  # MB per worker
SPATIAL_ORDERING = os.getenv("SPATIAL_ORDERING", "true").lower() == "true"  # Sort chunks spatially

# Spatial chunking parameters
SPATIAL_GRID_SIZE = int(os.getenv("SPATIAL_GRID_SIZE", "100"))  # Grid cells
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "10000"))

# Output parameters
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "geoparquet")
COMPRESSION = os.getenv("COMPRESSION", "snappy")
OUTPUT_COLUMNS = [
    PARCEL_ID_FIELD,
    PARCEL_ACREAGE_FIELD,
    "agriculture_pct",
    "developed_pct",
    "forest_pct",
    "other_pct",
    "rangeland_pasture_pct",
    "majority_land_use",
    "total_pixels",
    "valid_pixels",
    "geometry"
]

# Logging configuration
LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Quality control thresholds
MIN_VALID_PIXELS = float(os.getenv("MIN_VALID_PIXELS", "0.001"))  # Allow sub-pixel parcels
MAX_PROPORTION_ERROR = float(os.getenv("MAX_PROPORTION_ERROR", "0.01"))  # 1%

# Performance tuning
CACHE_SIZE_MB = int(os.getenv("CACHE_SIZE_MB", "1000"))
USE_MEMORY_MAPPING = os.getenv("USE_MEMORY_MAPPING", "true").lower() == "true"
ENABLE_PROGRESS_BAR = os.getenv("ENABLE_PROGRESS_BAR", "true").lower() == "true"

# Error handling
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "5"))
CONTINUE_ON_ERROR = os.getenv("CONTINUE_ON_ERROR", "true").lower() == "true"

def get_config() -> Dict[str, Any]:
    """Return configuration as a dictionary."""
    return {
        "data": {
            "raster_path": str(RASTER_PATH),
            "parcel_path": str(PARCEL_PATH),
            "parcel_id_field": PARCEL_ID_FIELD,
            "parcel_acreage_field": PARCEL_ACREAGE_FIELD,
        },
        "crs": {
            "raster": RASTER_CRS,
            "parcel": PARCEL_CRS,
            "output": OUTPUT_CRS,
        },
        "land_use": {
            "classes": LAND_USE_CLASSES,
            "descriptions": LAND_USE_DESCRIPTIONS,
        },
        "processing": {
            "chunk_size": CHUNK_SIZE,
            "raster_window_size": RASTER_WINDOW_SIZE,
            "max_memory_gb": MAX_MEMORY_GB,
            "n_workers": N_WORKERS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "output": {
            "format": OUTPUT_FORMAT,
            "compression": COMPRESSION,
            "columns": OUTPUT_COLUMNS,
            "dir": str(OUTPUT_DIR),
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT,
            "date_format": LOG_DATE_FORMAT,
            "dir": str(LOG_DIR),
        },
        "quality_control": {
            "min_valid_pixels": MIN_VALID_PIXELS,
            "max_proportion_error": MAX_PROPORTION_ERROR,
        },
        "performance": {
            "cache_size_mb": CACHE_SIZE_MB,
            "use_memory_mapping": USE_MEMORY_MAPPING,
            "enable_progress_bar": ENABLE_PROGRESS_BAR,
        },
        "error_handling": {
            "max_retries": MAX_RETRIES,
            "retry_delay_seconds": RETRY_DELAY_SECONDS,
            "continue_on_error": CONTINUE_ON_ERROR,
        },
    }

def setup_logging() -> None:
    """Set up logging configuration."""
    LOG_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(LOG_DIR / "pipeline.log"),
            logging.StreamHandler()
        ]
    )

# Create necessary directories
for dir_path in [OUTPUT_DIR, LOG_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)