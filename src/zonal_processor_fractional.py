"""Core zonal statistics processing with fractional pixel support."""
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm

from .config import (
    LAND_USE_CLASSES,
    PARCEL_ID_FIELD,
    PARCEL_ACREAGE_FIELD,
    MIN_VALID_PIXELS
)

logger = logging.getLogger(__name__)

class FractionalZonalStatsProcessor:
    """Calculate zonal statistics with fractional pixel coverage for accurate area calculations."""
    
    def __init__(self, n_workers: int = 1):
        """Initialize processor.
        
        Args:
            n_workers: Number of parallel workers
        """
        self.n_workers = n_workers
        self.processing_stats = {}
        
    def calculate_fractional_coverage(
        self,
        geometry,
        transform,
        shape
    ) -> np.ndarray:
        """Calculate fractional coverage of each pixel by the geometry.
        
        Args:
            geometry: Shapely geometry
            transform: Affine transform for the raster
            shape: Shape of the raster window (height, width)
            
        Returns:
            Array with fractional coverage values (0-1) for each pixel
        """
        # Create a higher resolution grid for sub-pixel calculation
        # Use 10x10 sub-pixels for each pixel (100 sub-pixels total)
        sub_pixel_factor = 10
        
        # Create sub-pixel transform
        sub_transform = transform * transform.scale(
            1/sub_pixel_factor,
            1/sub_pixel_factor
        )
        
        # Create sub-pixel shape
        sub_shape = (shape[0] * sub_pixel_factor, shape[1] * sub_pixel_factor)
        
        # Rasterize at sub-pixel resolution
        sub_pixel_mask = features.rasterize(
            [(geometry, 1)],
            out_shape=sub_shape,
            transform=sub_transform,
            fill=0,
            dtype=np.uint8
        )
        
        # Aggregate back to original pixel resolution
        # Reshape to group sub-pixels by their parent pixel
        reshaped = sub_pixel_mask.reshape(
            shape[0], sub_pixel_factor,
            shape[1], sub_pixel_factor
        )
        
        # Sum sub-pixels and divide by total to get fraction
        fractional_coverage = reshaped.sum(axis=(1, 3)) / (sub_pixel_factor ** 2)
        
        return fractional_coverage
    
    def calculate_land_use_proportions_fractional(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: Optional[int] = None,
        use_fractional: bool = True
    ) -> pd.DataFrame:
        """Calculate land use proportions with fractional pixel support.
        
        Args:
            parcels: Parcels GeoDataFrame  
            raster_path: Path to land use raster
            chunk_id: Optional chunk identifier for logging
            use_fractional: Whether to use fractional pixel counting
            
        Returns:
            DataFrame with land use proportions
        """
        chunk_str = f"[Chunk {chunk_id}] " if chunk_id else ""
        logger.info(f"{chunk_str}Processing {len(parcels)} parcels with fractional pixels")
        
        start_time = time.time()
        
        # Open raster to get bounds and transform
        with rasterio.open(raster_path) as src:
            # Get the bounds of the parcels
            parcel_bounds = parcels.total_bounds
            
            # Create window from bounds for efficiency
            window = rasterio.windows.from_bounds(
                *parcel_bounds,
                transform=src.transform
            )
            
            # Read the windowed raster data
            raster_data = src.read(1, window=window)
            window_transform = rasterio.windows.transform(window, src.transform)
            
            # Get pixel area in square meters
            pixel_width = abs(src.transform[0])
            pixel_height = abs(src.transform[4])
            pixel_area_m2 = pixel_width * pixel_height
            
            logger.info(f"{chunk_str}Loaded raster window: {raster_data.shape}")
            logger.info(f"{chunk_str}Pixel area: {pixel_area_m2:.2f} m²")
        
        # Process each parcel
        results = []
        
        for idx, parcel in parcels.iterrows():
            parcel_id = parcel[PARCEL_ID_FIELD]
            
            if use_fractional:
                # Calculate fractional coverage
                fractional_coverage = self.calculate_fractional_coverage(
                    parcel.geometry,
                    window_transform,
                    raster_data.shape
                )
                
                # Mask raster data with fractional coverage
                # Only include pixels with >0 coverage
                mask = fractional_coverage > 0
                
                if not mask.any():
                    # No pixels covered
                    result = self._empty_result(parcel)
                else:
                    # Get land use values for covered pixels
                    covered_values = raster_data[mask]
                    covered_fractions = fractional_coverage[mask]
                    
                    # Exclude NoData (0) values
                    valid_mask = covered_values != 0
                    
                    if not valid_mask.any():
                        result = self._empty_result(parcel)
                    else:
                        valid_values = covered_values[valid_mask]
                        valid_fractions = covered_fractions[valid_mask]
                        
                        # Calculate weighted land use areas
                        land_use_areas = {}
                        for class_id in LAND_USE_CLASSES.keys():
                            class_mask = valid_values == class_id
                            if class_mask.any():
                                # Sum fractional pixels for this class
                                land_use_areas[class_id] = valid_fractions[class_mask].sum()
                            else:
                                land_use_areas[class_id] = 0.0
                        
                        # Total fractional pixel count
                        total_fractional_pixels = sum(land_use_areas.values())
                        
                        # Calculate proportions
                        proportions = {}
                        for class_id, class_name in LAND_USE_CLASSES.items():
                            pct_name = f"{class_name.lower()}_pct"
                            if total_fractional_pixels > 0:
                                proportions[pct_name] = (land_use_areas[class_id] / total_fractional_pixels * 100)
                            else:
                                proportions[pct_name] = 0.0
                        
                        # Find majority class
                        if total_fractional_pixels > 0:
                            majority_class = max(land_use_areas.keys(), key=lambda k: land_use_areas[k])
                            majority_name = LAND_USE_CLASSES.get(majority_class, 'UNKNOWN')
                        else:
                            majority_name = 'UNKNOWN'
                        
                        # Calculate area in acres
                        total_area_m2 = total_fractional_pixels * pixel_area_m2
                        total_acres = total_area_m2 / 4046.86
                        
                        result = {
                            PARCEL_ID_FIELD: parcel_id,
                            **proportions,
                            'majority_land_use': majority_name,
                            'total_pixels': total_fractional_pixels,  # Now fractional
                            'valid_pixels': total_fractional_pixels,
                            'fractional_area_m2': total_area_m2,
                            'fractional_acres': total_acres,
                            'whole_pixels_covered': mask.sum(),  # Count of whole pixels touched
                        }
            else:
                # Use traditional whole-pixel counting
                result = self._calculate_whole_pixel_stats(
                    parcel, raster_data, window_transform, pixel_area_m2
                )
            
            # Add acreage if available
            if PARCEL_ACREAGE_FIELD in parcel.index:
                result[PARCEL_ACREAGE_FIELD] = parcel[PARCEL_ACREAGE_FIELD]
                
                # Calculate difference if using fractional
                if use_fractional and 'fractional_acres' in result:
                    result['acre_diff'] = result[PARCEL_ACREAGE_FIELD] - result['fractional_acres']
                    result['acre_diff_pct'] = (result['acre_diff'] / result[PARCEL_ACREAGE_FIELD] * 100) if result[PARCEL_ACREAGE_FIELD] > 0 else 0
            
            results.append(result)
        
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
            'parcels_per_second': parcels_per_second
        }
        
        return results_df
    
    def _empty_result(self, parcel) -> dict:
        """Create empty result for parcels with no coverage."""
        return {
            PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
            'agriculture_pct': 0.0,
            'developed_pct': 0.0,
            'forest_pct': 0.0,
            'other_pct': 0.0,
            'rangeland_pasture_pct': 0.0,
            'majority_land_use': 'UNKNOWN',
            'total_pixels': 0.0,
            'valid_pixels': 0.0,
            'fractional_area_m2': 0.0,
            'fractional_acres': 0.0,
            'whole_pixels_covered': 0,
        }
    
    def _calculate_whole_pixel_stats(
        self,
        parcel,
        raster_data,
        transform,
        pixel_area_m2
    ) -> dict:
        """Calculate statistics using whole pixel counting (traditional method)."""
        from rasterstats import zonal_stats
        
        stats_list = zonal_stats(
            [parcel.geometry],
            raster_data,
            affine=transform,
            categorical=True,
            nodata=0,
            all_touched=True
        )
        
        stats = stats_list[0] if stats_list else None
        
        if stats is None or not stats:
            return self._empty_result(parcel)
        
        # Calculate total pixels (excluding nodata)
        total_pixels = sum(stats.values())
        
        # Calculate proportions for each land use class
        proportions = {}
        for class_id, class_name in LAND_USE_CLASSES.items():
            pixel_count = stats.get(class_id, 0)
            pct_name = f"{class_name.lower()}_pct"
            proportions[pct_name] = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0.0
        
        # Find majority class
        if total_pixels > 0:
            majority_class = max(stats.keys(), key=lambda k: stats[k])
            majority_name = LAND_USE_CLASSES.get(majority_class, 'UNKNOWN')
        else:
            majority_name = 'UNKNOWN'
        
        return {
            PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
            **proportions,
            'majority_land_use': majority_name,
            'total_pixels': total_pixels,
            'valid_pixels': total_pixels,
            'whole_pixels_covered': total_pixels,
        }
    
    def process_chunk(
        self,
        parcel_chunk: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: int,
        use_fractional: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process a single chunk of parcels.
        
        Args:
            parcel_chunk: Chunk of parcels to process
            raster_path: Path to raster file
            chunk_id: Chunk identifier
            use_fractional: Whether to use fractional pixel counting
            
        Returns:
            Tuple of (results DataFrame, statistics dict)
        """
        try:
            results = self.calculate_land_use_proportions_fractional(
                parcel_chunk,
                raster_path,
                chunk_id,
                use_fractional
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