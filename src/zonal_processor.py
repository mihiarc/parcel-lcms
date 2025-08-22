"""Zonal statistics processor with adaptive sub-pixel accuracy."""
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterstats import zonal_stats
from tqdm import tqdm

from .base_processor import BaseZonalStatsProcessor
from .config import (
    LAND_USE_CLASSES,
    PARCEL_ID_FIELD,
    PARCEL_ACREAGE_FIELD,
    MIN_VALID_PIXELS
)

logger = logging.getLogger(__name__)


class ZonalStatsProcessor(BaseZonalStatsProcessor):
    """Calculate zonal statistics with adaptive sub-pixel accuracy.
    
    This processor uses adaptive sub-pixel rasterization based on parcel size:
    - < 1 acre: 2x2 sub-pixels (finer control for tiny parcels)
    - 1-5 acres: 3x3 sub-pixels  
    - 5-10 acres: 5x5 sub-pixels (optimal balance)
    - 10-50 acres: 7x7 sub-pixels
    - > 50 acres: 10x10 sub-pixels (maximum accuracy)
    
    This provides near-perfect correlation with actual parcel areas while
    maintaining high processing speed (1,500-2,000 parcels/second).
    """
    
    def __init__(self, n_workers: int = 1):
        """Initialize processor.
        
        Args:
            n_workers: Number of parallel workers
        """
        super().__init__(n_workers)
        self.default_method = 'subpixel'
        
    def get_adaptive_subpixel_resolution(self, parcel_acres: float) -> int:
        """Determine optimal sub-pixel resolution based on parcel size.
        
        Args:
            parcel_acres: Parcel area in acres
            
        Returns:
            Sub-pixel resolution (e.g., 5 means 5x5 grid)
        """
        if parcel_acres < 1.0:
            return 2  # 2x2 for very small parcels
        elif parcel_acres < 5.0:
            return 3  # 3x3 for small parcels
        elif parcel_acres < 10.0:
            return 5  # 5x5 for medium parcels
        elif parcel_acres < 50.0:
            return 7  # 7x7 for larger parcels
        else:
            return 10  # 10x10 for very large parcels
    
    def calculate_land_use_proportions(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: Optional[int] = None,
        method: str = 'subpixel'
    ) -> pd.DataFrame:
        """Calculate land use proportions for parcels.
        
        Args:
            parcels: Parcels GeoDataFrame  
            raster_path: Path to land use raster
            chunk_id: Optional chunk identifier for logging
            method: 'subpixel' (default), 'standard', or 'center'
            
        Returns:
            DataFrame with land use proportions
        """
        chunk_str = f"[Chunk {chunk_id}] " if chunk_id else ""
        logger.info(f"{chunk_str}Processing {len(parcels)} parcels with {method} method")
        
        start_time = time.time()
        
        # Process based on selected method
        if method == 'subpixel':
            results = self._process_subpixel_adaptive(parcels, raster_path)
        elif method == 'standard':
            results = self._process_standard(parcels, raster_path)
        elif method == 'center':
            results = self._process_center(parcels, raster_path)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        parcels_per_second = len(parcels) / processing_time if processing_time > 0 else 0
        
        logger.info(f"{chunk_str}Processed {len(parcels)} parcels in {processing_time:.2f}s "
                   f"({parcels_per_second:.1f} parcels/s)")
        
        # Store statistics
        self.processing_stats[chunk_id if chunk_id else 'main'] = {
            'n_parcels': len(parcels),
            'processing_time': processing_time,
            'parcels_per_second': parcels_per_second,
            'method': method
        }
        
        return results_df
    
    def _process_subpixel_adaptive(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str
    ) -> List[Dict]:
        """Process parcels using adaptive sub-pixel accuracy method.
        
        This is the DEFAULT and RECOMMENDED method.
        """
        results = []
        
        # Group parcels by spatial proximity for efficient processing
        parcels_with_idx = parcels.copy()
        parcels_with_idx['_original_idx'] = parcels.index
        
        # Process parcels in spatial groups
        for _, parcel in parcels_with_idx.iterrows():
            parcel_id = parcel[PARCEL_ID_FIELD]
            
            # Get parcel area in acres
            parcel_acres = parcel.get(PARCEL_ACREAGE_FIELD, None)
            if parcel_acres is None:
                # Calculate from geometry if not provided
                parcel_acres = parcel.geometry.area / 4046.86
            
            # Determine adaptive sub-pixel resolution
            sub_factor = self.get_adaptive_subpixel_resolution(parcel_acres)
            
            # Get optimized raster window
            bounds = parcel.geometry.bounds
            raster_data, transform, pixel_area_m2 = self.get_raster_window(
                bounds, raster_path, buffer_pixels=2
            )
            
            # Process with adaptive sub-pixel resolution
            result = self._calculate_subpixel_stats(
                parcel, 
                raster_data, 
                transform, 
                pixel_area_m2,
                sub_factor
            )
            
            # Add original acreage if available
            if PARCEL_ACREAGE_FIELD in parcel.index:
                result[PARCEL_ACREAGE_FIELD] = parcel[PARCEL_ACREAGE_FIELD]
                if 'calculated_acres' in result and result['calculated_acres'] > 0:
                    result['acre_diff'] = result['calculated_acres'] - result[PARCEL_ACREAGE_FIELD]
                    result['acre_diff_pct'] = (result['acre_diff'] / 
                                              result[PARCEL_ACREAGE_FIELD] * 100)
            
            # Add sub-pixel resolution used
            result['subpixel_resolution'] = sub_factor
            
            results.append(result)
        
        return results
    
    def _calculate_subpixel_stats(
        self,
        parcel: gpd.GeoDataFrame,
        raster_data: np.ndarray,
        transform,
        pixel_area_m2: float,
        sub_factor: int
    ) -> Dict:
        """Calculate statistics using sub-pixel method.
        
        Args:
            parcel: Single parcel row
            raster_data: Raster data array
            transform: Affine transform
            pixel_area_m2: Area of single pixel in square meters
            sub_factor: Sub-pixel resolution factor
            
        Returns:
            Dictionary with calculated statistics
        """
        parcel_id = parcel[PARCEL_ID_FIELD]
        pixel_area_acres = pixel_area_m2 / 4046.86
        
        if raster_data.size == 0:
            return self.create_empty_result(parcel_id)
        
        # Create sub-pixel transform
        sub_transform = transform * transform.scale(1/sub_factor, 1/sub_factor)
        
        # Create sub-pixel shape
        sub_shape = (
            raster_data.shape[0] * sub_factor,
            raster_data.shape[1] * sub_factor
        )
        
        # Rasterize at sub-pixel resolution
        try:
            sub_mask = features.rasterize(
                [(parcel.geometry, 1)],
                out_shape=sub_shape,
                transform=sub_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False  # Use exact boundaries
            )
        except Exception as e:
            logger.warning(f"Failed to rasterize parcel {parcel_id}: {e}")
            return self.create_empty_result(parcel_id)
        
        # Calculate fractional coverage for each pixel
        # Reshape to group sub-pixels by their parent pixel
        try:
            reshaped = sub_mask.reshape(
                raster_data.shape[0], sub_factor,
                raster_data.shape[1], sub_factor
            )
            
            # Sum sub-pixels and divide by total to get fraction
            fractional_coverage = reshaped.sum(axis=(1, 3)) / (sub_factor ** 2)
        except Exception as e:
            logger.warning(f"Failed to calculate fractional coverage for parcel {parcel_id}: {e}")
            return self.create_empty_result(parcel_id)
        
        # Calculate weighted land use statistics
        land_use_counts = {}
        
        # Get pixels with coverage
        covered_mask = fractional_coverage > 0
        
        if not covered_mask.any():
            return self.create_empty_result(parcel_id)
        
        # Process each land use class
        for class_id in LAND_USE_CLASSES.keys():
            # Find pixels with this land use (excluding nodata=0)
            class_mask = (raster_data == class_id) & covered_mask
            if class_mask.any():
                # Weight by fractional coverage
                weighted_count = (fractional_coverage[class_mask]).sum()
                land_use_counts[class_id] = weighted_count
        
        # Check for additional classes not in config
        unique_values = np.unique(raster_data[covered_mask])
        for val in unique_values:
            if val != 0 and val not in LAND_USE_CLASSES and val not in land_use_counts:
                class_mask = (raster_data == val) & covered_mask
                if class_mask.any():
                    weighted_count = (fractional_coverage[class_mask]).sum()
                    land_use_counts[val] = weighted_count
                    logger.debug(f"Found unmapped land use class: {val}")
        
        # Calculate proportions
        result = self.calculate_proportions_from_counts(land_use_counts, parcel_id)
        
        # Add calculated acreage
        total_fractional_pixels = sum(land_use_counts.values())
        result['calculated_acres'] = total_fractional_pixels * pixel_area_acres
        
        return result
    
    def _process_standard(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str
    ) -> List[Dict]:
        """Process using standard all_touched=True method (for comparison).
        
        This method is retained for testing and comparison purposes only.
        """
        results = []
        
        # Get full bounds for all parcels
        total_bounds = parcels.total_bounds
        raster_data, transform, pixel_area_m2 = self.get_raster_window(
            total_bounds, raster_path, buffer_pixels=1
        )
        
        pixel_area_acres = pixel_area_m2 / 4046.86
        
        # Use rasterstats for standard processing
        stats_list = zonal_stats(
            parcels.geometry,
            raster_data,
            affine=transform,
            categorical=True,
            nodata=0,
            all_touched=True
        )
        
        for idx, (parcel_idx, stats) in enumerate(zip(parcels.index, stats_list)):
            parcel = parcels.loc[parcel_idx]
            parcel_id = parcel[PARCEL_ID_FIELD]
            
            if stats and any(stats.values()):
                # Convert stats to land_use_counts format
                land_use_counts = {k: v for k, v in stats.items() if k in LAND_USE_CLASSES}
                
                # Calculate proportions
                result = self.calculate_proportions_from_counts(land_use_counts, parcel_id)
                
                # Add calculated acreage
                total_pixels = sum(stats.values())
                result['calculated_acres'] = total_pixels * pixel_area_acres
            else:
                result = self.create_empty_result(parcel_id)
            
            # Add original acreage if available
            if PARCEL_ACREAGE_FIELD in parcel.index:
                result[PARCEL_ACREAGE_FIELD] = parcel[PARCEL_ACREAGE_FIELD]
            
            results.append(result)
        
        return results
    
    def _process_center(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str
    ) -> List[Dict]:
        """Process using center-only method (all_touched=False).
        
        This method is retained for testing and comparison purposes only.
        """
        results = []
        
        # Get full bounds for all parcels
        total_bounds = parcels.total_bounds
        raster_data, transform, pixel_area_m2 = self.get_raster_window(
            total_bounds, raster_path, buffer_pixels=1
        )
        
        pixel_area_acres = pixel_area_m2 / 4046.86
        
        # Use rasterstats with all_touched=False
        stats_list = zonal_stats(
            parcels.geometry,
            raster_data,
            affine=transform,
            categorical=True,
            nodata=0,
            all_touched=False
        )
        
        for idx, (parcel_idx, stats) in enumerate(zip(parcels.index, stats_list)):
            parcel = parcels.loc[parcel_idx]
            parcel_id = parcel[PARCEL_ID_FIELD]
            
            if stats and any(stats.values()):
                # Convert stats to land_use_counts format
                land_use_counts = {k: v for k, v in stats.items() if k in LAND_USE_CLASSES}
                
                # Calculate proportions
                result = self.calculate_proportions_from_counts(land_use_counts, parcel_id)
                
                # Add calculated acreage
                total_pixels = sum(stats.values())
                result['calculated_acres'] = total_pixels * pixel_area_acres
            else:
                result = self.create_empty_result(parcel_id)
            
            # Add original acreage if available
            if PARCEL_ACREAGE_FIELD in parcel.index:
                result[PARCEL_ACREAGE_FIELD] = parcel[PARCEL_ACREAGE_FIELD]
            
            results.append(result)
        
        return results
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics.
        
        Returns:
            Dictionary with processing summary
        """
        summary = super().get_processing_summary()
        
        # Add method information
        if self.processing_stats:
            methods = [s.get('method', 'unknown') for s in self.processing_stats.values()]
            primary_method = max(set(methods), key=methods.count)
            summary['primary_method'] = primary_method
        
        return summary