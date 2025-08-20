"""Core zonal statistics processing using rasterstats."""
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm

from .config import (
    LAND_USE_CLASSES,
    PARCEL_ID_FIELD,
    PARCEL_ACREAGE_FIELD,
    MIN_VALID_PIXELS
)

logger = logging.getLogger(__name__)

class ZonalStatsProcessor:
    """Calculate zonal statistics for parcels against land use raster."""
    
    def __init__(self, n_workers: int = 1):
        """Initialize processor.
        
        Args:
            n_workers: Number of parallel workers
        """
        self.n_workers = n_workers
        self.processing_stats = {}
        
    def calculate_land_use_proportions(
        self,
        parcels: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: Optional[int] = None
    ) -> pd.DataFrame:
        """Calculate land use proportions for parcels.
        
        Args:
            parcels: Parcels GeoDataFrame  
            raster_path: Path to land use raster
            chunk_id: Optional chunk identifier for logging
            
        Returns:
            DataFrame with land use proportions
        """
        chunk_str = f"[Chunk {chunk_id}] " if chunk_id else ""
        logger.info(f"{chunk_str}Processing {len(parcels)} parcels")
        
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
            
            logger.info(f"{chunk_str}Loaded raster window: {raster_data.shape}")
        
        # Calculate zonal statistics
        logger.info(f"{chunk_str}Calculating zonal statistics")
        
        stats_list = zonal_stats(
            parcels.geometry,
            raster_data,
            affine=window_transform,
            categorical=True,
            nodata=0,
            all_touched=True  # Include all pixels that touch the parcel
        )
        
        # Process results into proportions
        results = []
        for idx, (parcel_idx, stats) in enumerate(zip(parcels.index, stats_list)):
            parcel_row = parcels.loc[parcel_idx]
            
            if stats is None or not stats:
                # No valid pixels for this parcel
                logger.warning(f"No valid pixels for parcel {parcel_row[PARCEL_ID_FIELD]}")
                result = {
                    PARCEL_ID_FIELD: parcel_row[PARCEL_ID_FIELD],
                    'agriculture_pct': 0.0,
                    'developed_pct': 0.0,
                    'forest_pct': 0.0,
                    'other_pct': 0.0,
                    'rangeland_pasture_pct': 0.0,
                    'majority_land_use': 'UNKNOWN',
                    'total_pixels': 0,
                    'valid_pixels': 0
                }
            else:
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
                
                result = {
                    PARCEL_ID_FIELD: parcel_row[PARCEL_ID_FIELD],
                    **proportions,
                    'majority_land_use': majority_name,
                    'total_pixels': total_pixels,
                    'valid_pixels': total_pixels  # Same as total since we exclude nodata
                }
            
            # Add acreage if available
            if PARCEL_ACREAGE_FIELD in parcel_row.index:
                result[PARCEL_ACREAGE_FIELD] = parcel_row[PARCEL_ACREAGE_FIELD]
            
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
    
    def process_chunk(
        self,
        parcel_chunk: gpd.GeoDataFrame,
        raster_path: str,
        chunk_id: int
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process a single chunk of parcels.
        
        Args:
            parcel_chunk: Chunk of parcels to process
            raster_path: Path to raster file
            chunk_id: Chunk identifier
            
        Returns:
            Tuple of (results DataFrame, statistics dict)
        """
        try:
            results = self.calculate_land_use_proportions(
                parcel_chunk,
                raster_path,
                chunk_id
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
            Validated DataFrame
        """
        logger.info("Validating results")
        
        # Check proportion sums
        proportion_cols = [
            'agriculture_pct', 'developed_pct', 'forest_pct',
            'other_pct', 'rangeland_pasture_pct'
        ]
        
        results_df['proportion_sum'] = results_df[proportion_cols].sum(axis=1)
        
        # Flag parcels with invalid proportion sums
        tolerance = 0.01  # 0.01% tolerance
        invalid_mask = abs(results_df['proportion_sum'] - 100.0) > tolerance
        
        # Only check parcels with valid pixels
        invalid_mask &= results_df['valid_pixels'] > 0
        
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            logger.warning(f"Found {n_invalid} parcels with invalid proportion sums")
            
            # Log examples
            examples = results_df[invalid_mask].head(5)
            logger.debug(f"Examples of invalid parcels:\n{examples}")
        
        # Check for parcels with no valid pixels
        no_pixels_mask = results_df['valid_pixels'] < MIN_VALID_PIXELS
        if no_pixels_mask.any():
            n_no_pixels = no_pixels_mask.sum()
            logger.warning(f"Found {n_no_pixels} parcels with < {MIN_VALID_PIXELS} valid pixels")
        
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
        total_time = sum(s['processing_time'] for s in self.processing_stats.values())
        avg_speed = total_parcels / total_time if total_time > 0 else 0
        
        return {
            'total_parcels': total_parcels,
            'total_time_seconds': total_time,
            'average_parcels_per_second': avg_speed,
            'n_chunks': len(self.processing_stats),
            'chunk_stats': self.processing_stats
        }