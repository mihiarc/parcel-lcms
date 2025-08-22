"""Base class for zonal statistics processors."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import windows
from rasterio.windows import Window

from .config import (
    LAND_USE_CLASSES,
    PARCEL_ID_FIELD,
    PARCEL_ACREAGE_FIELD,
    MIN_VALID_PIXELS
)

logger = logging.getLogger(__name__)


class BaseZonalStatsProcessor(ABC):
    """Abstract base class for all zonal statistics processors.
    
    Provides common functionality for:
    - Raster window management
    - Result validation
    - Processing statistics tracking
    - Error handling
    """
    
    def __init__(self, n_workers: int = 1):
        """Initialize base processor.
        
        Args:
            n_workers: Number of parallel workers
        """
        self.n_workers = n_workers
        self.processing_stats = {}
        self._raster_cache = {}
        self._cache_size_mb = 512  # Default cache size
        
    @abstractmethod
    def calculate_land_use_proportions(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate land use proportions for parcels.
        
        Args:
            parcels: Parcels GeoDataFrame  
            raster_path: Path to land use raster
            chunk_id: Optional chunk identifier for logging
            **kwargs: Additional method-specific parameters
            
        Returns:
            DataFrame with land use proportions
        """
        pass
    
    def get_raster_window(
        self,
        bounds: Tuple[float, float, float, float],
        raster_path: str,
        buffer_pixels: int = 2
    ) -> Tuple[np.ndarray, Any, float]:
        """Get optimized raster window for given bounds.
        
        Args:
            bounds: (minx, miny, maxx, maxy) bounds
            raster_path: Path to raster file
            buffer_pixels: Buffer in pixels around bounds
            
        Returns:
            Tuple of (raster_data, transform, pixel_area_m2)
        """
        # Create cache key
        cache_key = (bounds, raster_path, buffer_pixels)
        
        # Check cache
        if cache_key in self._raster_cache:
            return self._raster_cache[cache_key]
        
        with rasterio.open(raster_path) as src:
            # Create window from bounds
            window = windows.from_bounds(*bounds, transform=src.transform)
            
            # Round and buffer window
            col_off = max(0, int(window.col_off) - buffer_pixels)
            row_off = max(0, int(window.row_off) - buffer_pixels)
            width = min(int(window.width) + 2 * buffer_pixels + 1, src.width - col_off)
            height = min(int(window.height) + 2 * buffer_pixels + 1, src.height - row_off)
            
            # Create buffered window
            buffered_window = Window(col_off, row_off, width, height)
            
            # Read data
            raster_data = src.read(1, window=buffered_window)
            window_transform = windows.transform(buffered_window, src.transform)
            
            # Calculate pixel area
            pixel_width = abs(src.transform[0])
            pixel_height = abs(src.transform[4])
            pixel_area_m2 = pixel_width * pixel_height
            
        # Cache result (manage cache size)
        self._manage_cache(cache_key, (raster_data, window_transform, pixel_area_m2))
        
        return raster_data, window_transform, pixel_area_m2
    
    def _manage_cache(self, key: Any, value: Any) -> None:
        """Manage cache size by removing old entries if needed.
        
        Simple LRU-style cache management.
        """
        # Estimate size of cached data (rough approximation)
        if isinstance(value[0], np.ndarray):
            size_mb = value[0].nbytes / (1024 * 1024)
            
            # Check total cache size
            total_size_mb = sum(
                v[0].nbytes / (1024 * 1024)
                for v in self._raster_cache.values()
                if isinstance(v[0], np.ndarray)
            )
            
            # Remove oldest entries if cache is too large
            while total_size_mb + size_mb > self._cache_size_mb and self._raster_cache:
                # Remove first (oldest) entry
                oldest_key = next(iter(self._raster_cache))
                del self._raster_cache[oldest_key]
                total_size_mb = sum(
                    v[0].nbytes / (1024 * 1024)
                    for v in self._raster_cache.values()
                    if isinstance(v[0], np.ndarray)
                )
        
        # Add to cache
        self._raster_cache[key] = value
    
    def create_empty_result(self, parcel_id: Any) -> Dict:
        """Create empty result for parcels with no valid pixels.
        
        Args:
            parcel_id: Parcel identifier
            
        Returns:
            Dictionary with zero values for all land use classes
        """
        result = {
            PARCEL_ID_FIELD: parcel_id,
            'agriculture_pct': 0.0,
            'developed_pct': 0.0,
            'forest_pct': 0.0,
            'other_pct': 0.0,
            'rangeland_pasture_pct': 0.0,
            'majority_land_use': 'NO_DATA',
            'total_pixels': 0.0,
            'valid_pixels': 0.0,
            'calculated_acres': 0.0,
        }
        return result
    
    def calculate_proportions_from_counts(
        self,
        land_use_counts: Dict[int, float],
        parcel_id: Any
    ) -> Dict:
        """Calculate land use proportions from pixel counts.
        
        Args:
            land_use_counts: Dictionary mapping land use class to pixel count
            parcel_id: Parcel identifier
            
        Returns:
            Dictionary with calculated proportions and statistics
        """
        total_pixels = sum(land_use_counts.values())
        
        if total_pixels == 0:
            return self.create_empty_result(parcel_id)
        
        # Calculate proportions
        proportions = {}
        for class_id, class_name in LAND_USE_CLASSES.items():
            pct_name = f"{class_name.lower()}_pct"
            count = land_use_counts.get(class_id, 0)
            proportions[pct_name] = (count / total_pixels * 100)
        
        # Find majority class
        if land_use_counts:
            majority_class = max(land_use_counts.keys(), key=lambda k: land_use_counts[k])
            majority_name = LAND_USE_CLASSES.get(majority_class, f'CLASS_{majority_class}')
        else:
            majority_name = 'NO_DATA'
        
        result = {
            PARCEL_ID_FIELD: parcel_id,
            **proportions,
            'majority_land_use': majority_name,
            'total_pixels': total_pixels,
            'valid_pixels': total_pixels,
        }
        
        return result
    
    def process_chunk(
        self,
        parcel_chunk: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process a single chunk of parcels.
        
        Args:
            parcel_chunk: Chunk of parcels to process
            raster_path: Path to raster file
            chunk_id: Chunk identifier
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (results DataFrame, statistics dict)
        """
        try:
            results = self.calculate_land_use_proportions(
                parcel_chunk,
                raster_path,
                chunk_id,
                **kwargs
            )
            
            stats = {
                'chunk_id': chunk_id,
                'n_parcels': len(parcel_chunk),
                'n_processed': len(results),
                'status': 'success'
            }
            
            return results, stats
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            
            # Return empty results with error stats
            empty_results = pd.DataFrame()
            stats = {
                'chunk_id': chunk_id,
                'n_parcels': len(parcel_chunk),
                'n_processed': 0,
                'status': 'error',
                'error': str(e)
            }
            
            return empty_results, stats
    
    def validate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Validate zonal statistics results.
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Validated DataFrame with is_valid flag
        """
        logger.info("Validating results")
        
        # Check proportion sums
        proportion_cols = [
            'agriculture_pct', 'developed_pct', 'forest_pct',
            'other_pct', 'rangeland_pasture_pct'
        ]
        
        # Only validate columns that exist
        existing_cols = [col for col in proportion_cols if col in results_df.columns]
        
        if existing_cols:
            results_df['proportion_sum'] = results_df[existing_cols].sum(axis=1)
            
            # Flag parcels with invalid proportion sums (allowing small rounding errors)
            tolerance = 0.1  # 0.1% tolerance for rounding
            invalid_mask = abs(results_df['proportion_sum'] - 100.0) > tolerance
            
            # Only check parcels with valid pixels
            invalid_mask &= results_df['valid_pixels'] > 0
            
            if invalid_mask.any():
                n_invalid = invalid_mask.sum()
                logger.warning(f"Found {n_invalid} parcels with proportion sums != 100%")
                
                # Log examples
                examples = results_df[invalid_mask].head(3)
                for idx, row in examples.iterrows():
                    logger.debug(f"Parcel {row[PARCEL_ID_FIELD]}: sum={row['proportion_sum']:.2f}%")
        else:
            invalid_mask = pd.Series([False] * len(results_df))
        
        # Check for parcels with no valid pixels
        no_pixels_mask = results_df['valid_pixels'] < MIN_VALID_PIXELS
        if no_pixels_mask.any():
            n_no_pixels = no_pixels_mask.sum()
            logger.info(f"Found {n_no_pixels} parcels with < {MIN_VALID_PIXELS} valid pixels")
        
        # Add validation flag
        results_df['is_valid'] = ~invalid_mask & ~no_pixels_mask
        
        # Log validation summary
        logger.info(f"Validation complete: {results_df['is_valid'].sum()}/{len(results_df)} valid parcels")
        
        return results_df
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics.
        
        Returns:
            Dictionary with processing summary
        """
        if not self.processing_stats:
            return {}
        
        total_parcels = sum(s['n_parcels'] for s in self.processing_stats.values())
        total_time = sum(s.get('processing_time', 0) for s in self.processing_stats.values())
        avg_speed = total_parcels / total_time if total_time > 0 else 0
        
        return {
            'total_parcels': total_parcels,
            'total_time_seconds': total_time,
            'average_parcels_per_second': avg_speed,
            'n_chunks': len(self.processing_stats),
            'chunk_stats': self.processing_stats
        }
    
    def clear_cache(self) -> None:
        """Clear the raster cache to free memory."""
        self._raster_cache.clear()
        logger.debug("Cleared raster cache")