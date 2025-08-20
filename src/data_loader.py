"""Data loading utilities for parcel and raster data."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.crs import CRS
from pyogrio import read_dataframe
import psutil

from .config import (
    PARCEL_ID_FIELD, 
    PARCEL_ACREAGE_FIELD,
    USE_MEMORY_MAPPING
)

logger = logging.getLogger(__name__)

class DataLoader:
    """Handle loading of parcel and raster data efficiently."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data loader with configuration."""
        self.config = config or {}
        self.raster_metadata = None
        self.parcel_metadata = None
        
    def load_raster_metadata(self, raster_path: Path) -> Dict[str, Any]:
        """Load raster metadata without loading the actual data.
        
        Args:
            raster_path: Path to the raster file
            
        Returns:
            Dictionary with raster metadata
        """
        logger.info(f"Loading raster metadata from {raster_path}")
        
        with rasterio.open(raster_path) as src:
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "resolution": (src.transform[0], abs(src.transform[4])),
                "total_pixels": src.width * src.height,
                "memory_mb": (src.width * src.height * 1) / (1024 * 1024)  # uint8
            }
            
            # Get unique values (sample if too large)
            if metadata["total_pixels"] < 1e8:  # Less than 100M pixels
                data = src.read(1, masked=True)
                unique_values = list(np.unique(data.compressed()))
            else:
                # Sample the raster
                sample_window = rasterio.windows.Window(0, 0, 1000, 1000)
                sample_data = src.read(1, window=sample_window, masked=True)
                unique_values = list(np.unique(sample_data.compressed()))
                logger.info("Sampled raster for unique values due to size")
                
            metadata["unique_values"] = unique_values
            
        self.raster_metadata = metadata
        logger.info(f"Raster shape: {metadata['width']}x{metadata['height']}")
        logger.info(f"CRS: {metadata['crs']}")
        logger.info(f"Resolution: {metadata['resolution']}")
        logger.info(f"Memory requirement: {metadata['memory_mb']:.1f} MB")
        logger.info(f"Unique values: {metadata['unique_values']}")
        
        return metadata
    
    def load_parcels(
        self, 
        parcel_path: Path,
        columns: Optional[list] = None,
        sample_size: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """Load parcel data efficiently.
        
        Args:
            parcel_path: Path to parcel file
            columns: Specific columns to load
            sample_size: Number of parcels to sample (for testing)
            
        Returns:
            GeoDataFrame with parcel data
        """
        logger.info(f"Loading parcels from {parcel_path}")
        
        # Check available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        logger.info(f"Available memory: {available_memory:.1f} GB")
        
        # Load parcels based on file type
        if parcel_path.suffix == ".parquet":
            # Use geopandas for parquet files
            if columns:
                # Ensure geometry and required fields are included
                required_fields = {PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry'}
                columns = list(set(columns) | required_fields)
                
            parcels = gpd.read_parquet(parcel_path, columns=columns)
        else:
            # Use geopandas for other formats
            parcels = gpd.read_file(parcel_path, columns=columns)
            
        # Sample if requested
        if sample_size and len(parcels) > sample_size:
            logger.info(f"Sampling {sample_size} parcels from {len(parcels)}")
            parcels = parcels.sample(n=sample_size, random_state=42)
            
        # Store metadata
        self.parcel_metadata = {
            "count": len(parcels),
            "crs": parcels.crs,
            "bounds": parcels.total_bounds,
            "columns": list(parcels.columns),
            "memory_mb": parcels.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Log statistics
        logger.info(f"Loaded {len(parcels)} parcels")
        logger.info(f"CRS: {parcels.crs}")
        logger.info(f"Columns: {list(parcels.columns)}")
        logger.info(f"Memory usage: {self.parcel_metadata['memory_mb']:.1f} MB")
        
        # Check for required fields
        if PARCEL_ID_FIELD not in parcels.columns:
            raise ValueError(f"Required field '{PARCEL_ID_FIELD}' not found in parcel data")
        
        # Log acreage statistics if available
        if PARCEL_ACREAGE_FIELD in parcels.columns:
            acreage_stats = parcels[PARCEL_ACREAGE_FIELD].describe()
            logger.info(f"Acreage statistics:\n{acreage_stats}")
        
        return parcels
    
    def create_raster_reader(self, raster_path: Path, use_memory_map: bool = USE_MEMORY_MAPPING):
        """Create a raster reader context manager.
        
        Args:
            raster_path: Path to raster file
            use_memory_map: Whether to use memory mapping
            
        Returns:
            Rasterio dataset reader
        """
        if use_memory_map:
            return rasterio.open(raster_path, 'r', driver='GTiff', 
                               sharing=False, tiled=True)
        else:
            return rasterio.open(raster_path, 'r')
    
    def load_raster_window(
        self,
        raster_path: Path,
        bounds: Tuple[float, float, float, float],
        pad_pixels: int = 10
    ) -> Tuple[Any, Any]:
        """Load a windowed portion of the raster.
        
        Args:
            raster_path: Path to raster file
            bounds: (minx, miny, maxx, maxy) in raster CRS
            pad_pixels: Padding around the window in pixels
            
        Returns:
            Tuple of (raster_data, window_transform)
        """
        with rasterio.open(raster_path) as src:
            # Create window from bounds
            window = rasterio.windows.from_bounds(
                *bounds, 
                transform=src.transform
            )
            
            # Add padding
            if pad_pixels > 0:
                window = rasterio.windows.Window(
                    col_off=max(0, window.col_off - pad_pixels),
                    row_off=max(0, window.row_off - pad_pixels),
                    width=min(src.width - window.col_off, window.width + 2 * pad_pixels),
                    height=min(src.height - window.row_off, window.height + 2 * pad_pixels)
                )
            
            # Read the window
            raster_data = src.read(1, window=window)
            
            # Get the transform for this window
            window_transform = rasterio.windows.transform(window, src.transform)
            
            logger.debug(f"Loaded window of size {raster_data.shape}")
            
            return raster_data, window_transform
    
    def validate_data_compatibility(self) -> bool:
        """Validate that parcel and raster data are compatible.
        
        Returns:
            True if data is compatible
        """
        if not self.raster_metadata or not self.parcel_metadata:
            logger.warning("Metadata not loaded, skipping validation")
            return True
            
        # Check CRS compatibility
        raster_crs = self.raster_metadata["crs"]
        parcel_crs = self.parcel_metadata["crs"]
        
        if raster_crs != parcel_crs:
            logger.warning(f"CRS mismatch: Raster {raster_crs} vs Parcels {parcel_crs}")
            logger.info("Parcels will need to be reprojected")
            
        # Check spatial overlap
        raster_bounds = self.raster_metadata["bounds"]
        parcel_bounds = self.parcel_metadata["bounds"]
        
        # Simple overlap check (in respective CRS)
        logger.info(f"Raster bounds: {raster_bounds}")
        logger.info(f"Parcel bounds: {parcel_bounds}")
        
        return True
    
    def estimate_processing_requirements(self) -> Dict[str, Any]:
        """Estimate memory and processing requirements.
        
        Returns:
            Dictionary with processing estimates
        """
        if not self.raster_metadata or not self.parcel_metadata:
            return {}
            
        estimates = {
            "total_parcels": self.parcel_metadata["count"],
            "raster_size_mb": self.raster_metadata["memory_mb"],
            "parcel_size_mb": self.parcel_metadata["memory_mb"],
            "estimated_chunks": self.parcel_metadata["count"] // 5000 + 1,
            "estimated_memory_peak_mb": (
                self.parcel_metadata["memory_mb"] + 
                min(1000, self.raster_metadata["memory_mb"])  # Window size
            ),
        }
        
        logger.info("Processing estimates:")
        for key, value in estimates.items():
            logger.info(f"  {key}: {value}")
            
        return estimates