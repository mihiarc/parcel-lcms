"""Data preprocessing utilities for CRS transformation and validation."""
import logging
from typing import Optional, Tuple
import warnings

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.validation import make_valid

from .config import PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handle CRS transformation and geometry validation."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.transformer = None
        self.stats = {}
        
    def transform_parcels_crs(
        self, 
        parcels: gpd.GeoDataFrame,
        target_crs: str,
        validate_geometries: bool = True
    ) -> gpd.GeoDataFrame:
        """Transform parcels to target CRS.
        
        Args:
            parcels: Input parcels GeoDataFrame
            target_crs: Target coordinate reference system
            validate_geometries: Whether to validate geometries after transformation
            
        Returns:
            Transformed GeoDataFrame
        """
        logger.info(f"Transforming parcels from {parcels.crs} to {target_crs}")
        
        # Store original CRS
        original_crs = parcels.crs
        
        # Transform to target CRS
        parcels_transformed = parcels.to_crs(target_crs)
        
        # Validate if requested
        if validate_geometries:
            parcels_transformed = self.validate_geometries(parcels_transformed)
        
        # Log transformation statistics
        logger.info(f"Transformed {len(parcels_transformed)} parcels")
        logger.info(f"New bounds: {parcels_transformed.total_bounds}")
        
        # Check for any issues
        if parcels_transformed.geometry.is_empty.any():
            n_empty = parcels_transformed.geometry.is_empty.sum()
            logger.warning(f"Found {n_empty} empty geometries after transformation")
            
        return parcels_transformed
    
    def validate_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate and repair geometries.
        
        Args:
            gdf: GeoDataFrame with geometries to validate
            
        Returns:
            GeoDataFrame with valid geometries
        """
        logger.info("Validating geometries")
        
        # Check for invalid geometries
        invalid_mask = ~gdf.geometry.is_valid
        n_invalid = invalid_mask.sum()
        
        if n_invalid > 0:
            logger.warning(f"Found {n_invalid} invalid geometries, attempting repair")
            
            # Repair invalid geometries
            gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].apply(
                lambda geom: make_valid(geom)
            )
            
            # Check again
            still_invalid = ~gdf.geometry.is_valid
            if still_invalid.any():
                logger.error(f"{still_invalid.sum()} geometries still invalid after repair")
                # Option: remove invalid geometries or handle differently
                gdf = gdf[gdf.geometry.is_valid].copy()
        
        # Check for empty geometries
        empty_mask = gdf.geometry.is_empty
        n_empty = empty_mask.sum()
        
        if n_empty > 0:
            logger.warning(f"Found {n_empty} empty geometries, removing")
            gdf = gdf[~empty_mask].copy()
        
        # Check for null geometries
        null_mask = gdf.geometry.isna()
        n_null = null_mask.sum()
        
        if n_null > 0:
            logger.warning(f"Found {n_null} null geometries, removing")
            gdf = gdf[~null_mask].copy()
        
        # Store validation statistics
        self.stats['validation'] = {
            'n_invalid': n_invalid,
            'n_empty': n_empty,
            'n_null': n_null,
            'n_valid': len(gdf)
        }
        
        logger.info(f"Validation complete: {len(gdf)} valid geometries")
        
        return gdf
    
    def clip_to_raster_extent(
        self,
        parcels: gpd.GeoDataFrame,
        raster_bounds: Tuple[float, float, float, float],
        buffer_m: float = 0
    ) -> gpd.GeoDataFrame:
        """Clip parcels to raster extent.
        
        Args:
            parcels: Parcels GeoDataFrame
            raster_bounds: Raster bounds (minx, miny, maxx, maxy)
            buffer_m: Buffer distance in meters
            
        Returns:
            Clipped GeoDataFrame
        """
        logger.info(f"Clipping parcels to raster extent: {raster_bounds}")
        
        # Create bounding box geometry
        bbox = box(*raster_bounds)
        
        # Apply buffer if specified
        if buffer_m > 0:
            bbox = bbox.buffer(buffer_m)
            logger.info(f"Applied {buffer_m}m buffer to bounds")
        
        # Clip parcels
        n_before = len(parcels)
        parcels_clipped = parcels[parcels.geometry.intersects(bbox)].copy()
        n_after = len(parcels_clipped)
        
        logger.info(f"Clipped {n_before - n_after} parcels outside raster extent")
        logger.info(f"Remaining parcels: {n_after}")
        
        # Store statistics
        self.stats['clipping'] = {
            'n_before': n_before,
            'n_after': n_after,
            'n_removed': n_before - n_after
        }
        
        return parcels_clipped
    
    def filter_parcels_by_area(
        self,
        parcels: gpd.GeoDataFrame,
        min_area_m2: float = 0,
        max_area_m2: Optional[float] = None
    ) -> gpd.GeoDataFrame:
        """Filter parcels by area threshold.
        
        Args:
            parcels: Parcels GeoDataFrame
            min_area_m2: Minimum area in square meters
            max_area_m2: Maximum area in square meters
            
        Returns:
            Filtered GeoDataFrame
        """
        logger.info("Filtering parcels by area")
        
        # Calculate areas if not present
        if 'area_m2' not in parcels.columns:
            parcels['area_m2'] = parcels.geometry.area
        
        n_before = len(parcels)
        
        # Apply filters
        mask = parcels['area_m2'] >= min_area_m2
        if max_area_m2:
            mask &= parcels['area_m2'] <= max_area_m2
        
        parcels_filtered = parcels[mask].copy()
        n_after = len(parcels_filtered)
        
        logger.info(f"Filtered {n_before - n_after} parcels by area")
        max_area_str = f"{max_area_m2:.1f}" if max_area_m2 else "inf"
        logger.info(f"Area range: {min_area_m2:.1f} - {max_area_str} m²")
        logger.info(f"Remaining parcels: {n_after}")
        
        return parcels_filtered
    
    def add_spatial_index(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add spatial index to GeoDataFrame for faster operations.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with spatial index
        """
        logger.info("Building spatial index")
        
        # Create spatial index
        gdf.sindex
        
        logger.info("Spatial index created")
        
        return gdf
    
    def prepare_for_processing(
        self,
        parcels: gpd.GeoDataFrame,
        target_crs: str,
        raster_bounds: Optional[Tuple] = None,
        min_area_m2: float = 0
    ) -> gpd.GeoDataFrame:
        """Complete preprocessing pipeline.
        
        Args:
            parcels: Input parcels
            target_crs: Target CRS for transformation
            raster_bounds: Optional raster bounds for clipping
            min_area_m2: Minimum parcel area
            
        Returns:
            Preprocessed parcels ready for zonal statistics
        """
        logger.info("Starting preprocessing pipeline")
        
        # Transform CRS
        parcels = self.transform_parcels_crs(parcels, target_crs)
        
        # Validate geometries
        parcels = self.validate_geometries(parcels)
        
        # Clip to raster extent if bounds provided
        if raster_bounds:
            parcels = self.clip_to_raster_extent(parcels, raster_bounds)
        
        # Filter by area
        if min_area_m2 > 0:
            parcels = self.filter_parcels_by_area(parcels, min_area_m2)
        
        # Add spatial index
        parcels = self.add_spatial_index(parcels)
        
        # Add processing ID if not present
        if 'processing_id' not in parcels.columns:
            parcels['processing_id'] = range(len(parcels))
        
        logger.info(f"Preprocessing complete: {len(parcels)} parcels ready")
        
        return parcels
    
    def get_preprocessing_stats(self) -> dict:
        """Get preprocessing statistics.
        
        Returns:
            Dictionary with preprocessing statistics
        """
        return self.stats