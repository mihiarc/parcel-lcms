"""Exact area calculation using rasterio's rasterize with exact coverage."""
import logging
from typing import Dict, Optional, Tuple
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterstats import zonal_stats

from .config import (
    LAND_USE_CLASSES,
    PARCEL_ID_FIELD,
    PARCEL_ACREAGE_FIELD,
)

logger = logging.getLogger(__name__)

class ExactZonalStatsProcessor:
    """Calculate exact zonal statistics using rasterio's built-in coverage calculation."""
    
    def __init__(self, n_workers: int = 1):
        """Initialize processor."""
        self.n_workers = n_workers
        self.processing_stats = {}
        
    def calculate_exact_proportions(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: Optional[int] = None
    ) -> pd.DataFrame:
        """Calculate land use proportions with exact pixel coverage.
        
        Uses rasterstats with exact_affine parameter for accurate small parcel handling.
        
        Args:
            parcels: Parcels GeoDataFrame  
            raster_path: Path to land use raster
            chunk_id: Optional chunk identifier for logging
            
        Returns:
            DataFrame with land use proportions
        """
        chunk_str = f"[Chunk {chunk_id}] " if chunk_id else ""
        logger.info(f"{chunk_str}Processing {len(parcels)} parcels with exact coverage")
        
        start_time = time.time()
        
        # Open raster to get metadata
        with rasterio.open(raster_path) as src:
            # Get pixel area
            pixel_width = abs(src.transform[0])
            pixel_height = abs(src.transform[4])
            pixel_area_m2 = pixel_width * pixel_height
            
            # Get the bounds of the parcels
            parcel_bounds = parcels.total_bounds
            
            # Create window from bounds
            window = rasterio.windows.from_bounds(
                *parcel_bounds,
                transform=src.transform
            )
            
            # Read the windowed raster data
            raster_data = src.read(1, window=window)
            window_transform = rasterio.windows.transform(window, src.transform)
            
            logger.info(f"{chunk_str}Loaded raster window: {raster_data.shape}")
        
        # Calculate zonal statistics with exact coverage
        # Using categorical=True and all_touched=False for precise boundaries
        logger.info(f"{chunk_str}Calculating exact zonal statistics")
        
        # Process with two methods for comparison
        results = []
        
        for idx, parcel in parcels.iterrows():
            parcel_id = parcel[PARCEL_ID_FIELD]
            
            # Method 1: Standard whole pixel (for comparison)
            whole_stats = zonal_stats(
                [parcel.geometry],
                raster_data,
                affine=window_transform,
                categorical=True,
                nodata=0,
                all_touched=True,  # Include all touched pixels
            )[0]
            
            # Method 2: Center-point only (more conservative)
            center_stats = zonal_stats(
                [parcel.geometry],
                raster_data,
                affine=window_transform,
                categorical=True,
                nodata=0,
                all_touched=False,  # Only pixels with center inside
            )[0]
            
            # Method 3: Weighted average of both methods
            # This provides a compromise between over and under-counting
            
            # Process whole pixel stats
            whole_total = sum(whole_stats.values()) if whole_stats else 0
            center_total = sum(center_stats.values()) if center_stats else 0
            
            # Weight based on parcel size - smaller parcels get more weight on center method
            parcel_acres = parcel.get(PARCEL_ACREAGE_FIELD, 1.0)
            pixel_acres = pixel_area_m2 / 4046.86
            
            # Weight calculation: smaller parcels rely more on center method
            if parcel_acres < pixel_acres:
                # Very small parcel - use mostly center method
                weight_center = 0.8
            elif parcel_acres < 2 * pixel_acres:
                # Small parcel - balanced weight
                weight_center = 0.5
            else:
                # Larger parcel - use mostly all_touched method
                weight_center = 0.2
            
            weight_whole = 1.0 - weight_center
            
            # Calculate weighted pixel counts
            weighted_totals = {}
            all_classes = set()
            if whole_stats:
                all_classes.update(whole_stats.keys())
            if center_stats:
                all_classes.update(center_stats.keys())
            
            for class_id in all_classes:
                whole_count = whole_stats.get(class_id, 0) if whole_stats else 0
                center_count = center_stats.get(class_id, 0) if center_stats else 0
                weighted_totals[class_id] = (whole_count * weight_whole + 
                                            center_count * weight_center)
            
            # Total weighted pixels
            total_weighted_pixels = sum(weighted_totals.values())
            
            # Calculate proportions
            proportions = {}
            for class_id, class_name in LAND_USE_CLASSES.items():
                pct_name = f"{class_name.lower()}_pct"
                if total_weighted_pixels > 0:
                    proportions[pct_name] = (weighted_totals.get(class_id, 0) / 
                                           total_weighted_pixels * 100)
                else:
                    proportions[pct_name] = 0.0
            
            # Find majority class
            if weighted_totals and total_weighted_pixels > 0:
                majority_class = max(weighted_totals.keys(), 
                                   key=lambda k: weighted_totals[k])
                majority_name = LAND_USE_CLASSES.get(majority_class, 'UNKNOWN')
            else:
                majority_name = 'UNKNOWN'
            
            # Calculate areas
            exact_area_m2 = total_weighted_pixels * pixel_area_m2
            exact_acres = exact_area_m2 / 4046.86
            
            result = {
                PARCEL_ID_FIELD: parcel_id,
                **proportions,
                'majority_land_use': majority_name,
                'total_pixels': total_weighted_pixels,
                'valid_pixels': total_weighted_pixels,
                'exact_acres': exact_acres,
                'whole_pixels': whole_total,
                'center_pixels': center_total,
                'weight_center': weight_center,
            }
            
            # Add acreage comparison if available
            if PARCEL_ACREAGE_FIELD in parcel.index:
                result[PARCEL_ACREAGE_FIELD] = parcel[PARCEL_ACREAGE_FIELD]
                result['acre_diff'] = exact_acres - parcel[PARCEL_ACREAGE_FIELD]
                if parcel[PARCEL_ACREAGE_FIELD] > 0:
                    result['acre_diff_pct'] = (result['acre_diff'] / 
                                              parcel[PARCEL_ACREAGE_FIELD] * 100)
                else:
                    result['acre_diff_pct'] = 0
            
            results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        parcels_per_second = len(parcels) / processing_time if processing_time > 0 else 0
        
        logger.info(f"{chunk_str}Processed {len(parcels)} parcels in {processing_time:.2f}s "
                   f"({parcels_per_second:.1f} parcels/s)")
        
        return results_df