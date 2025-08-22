"""Parallel chunk processing for zonal statistics."""
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing as mp
import psutil
from pathlib import Path

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from .zonal_processor import ZonalStatsProcessor
from .config import N_WORKERS, CHUNK_SIZE

logger = logging.getLogger(__name__)


def process_chunk_worker(
    chunk_data: Tuple[int, gpd.GeoDataFrame],
    raster_path: str,
    method: str = 'subpixel',
    config_dict: Optional[Dict] = None
) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    """Worker function for processing a single chunk.
    
    This function runs in a separate process and must be pickleable.
    
    Args:
        chunk_data: Tuple of (chunk_id, chunk GeoDataFrame)
        raster_path: Path to raster file
        method: Processing method
        config_dict: Optional config overrides
        
    Returns:
        Tuple of (chunk_id, results DataFrame, statistics dict)
    """
    chunk_id, chunk = chunk_data
    
    # Configure GDAL for this worker process
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    os.environ['VSI_CACHE'] = 'TRUE'
    os.environ['VSI_CACHE_SIZE'] = '500000000'  # 500MB per worker
    
    # Create processor instance for this worker
    processor = ZonalStatsProcessor(n_workers=1)
    
    try:
        # Process the chunk
        results_df = processor.calculate_land_use_proportions(
            chunk,
            raster_path,
            chunk_id=chunk_id,
            method=method
        )
        
        # Get processing statistics
        stats = {
            'chunk_id': chunk_id,
            'n_parcels': len(chunk),
            'n_processed': len(results_df),
            'status': 'success',
            'worker_pid': os.getpid()
        }
        
        # Clear processor cache to free memory
        processor.clear_cache()
        
        return chunk_id, results_df, stats
        
    except Exception as e:
        logger.error(f"Worker error processing chunk {chunk_id}: {str(e)}")
        
        # Return empty results with error stats
        empty_df = pd.DataFrame()
        error_stats = {
            'chunk_id': chunk_id,
            'n_parcels': len(chunk),
            'n_processed': 0,
            'status': 'error',
            'error': str(e),
            'worker_pid': os.getpid()
        }
        
        return chunk_id, empty_df, error_stats


class ParallelChunkProcessor:
    """Manages parallel processing of chunks with memory management."""
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        max_memory_gb: float = 16.0,
        enable_progress: bool = True
    ):
        """Initialize parallel processor.
        
        Args:
            n_workers: Number of worker processes (default: N_WORKERS from config)
            max_memory_gb: Maximum memory usage in GB
            enable_progress: Show progress bar
        """
        self.n_workers = n_workers or N_WORKERS
        self.max_memory_gb = max_memory_gb
        self.enable_progress = enable_progress
        
        # Adjust workers based on available memory
        self._adjust_workers_for_memory()
        
        logger.info(f"Initialized parallel processor with {self.n_workers} workers")
        
    def _adjust_workers_for_memory(self):
        """Adjust number of workers based on available memory."""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate memory per worker (2GB for raster cache + processing)
        memory_per_worker = 2.0
        
        # Calculate max workers based on memory
        max_workers_by_memory = int(min(
            available_memory_gb / memory_per_worker,
            self.max_memory_gb / memory_per_worker
        ))
        
        # Use the minimum of configured and memory-limited workers
        if max_workers_by_memory < self.n_workers:
            logger.warning(f"Reducing workers from {self.n_workers} to {max_workers_by_memory} due to memory constraints")
            self.n_workers = max(1, max_workers_by_memory)
    
    def process_chunks_parallel(
        self,
        chunks: List[Tuple[int, gpd.GeoDataFrame]],
        raster_path: str,
        method: str = 'subpixel',
        checkpoint_callback: Optional[callable] = None
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """Process multiple chunks in parallel.
        
        Args:
            chunks: List of (chunk_id, chunk_gdf) tuples
            raster_path: Path to raster file
            method: Processing method
            checkpoint_callback: Optional callback for saving checkpoints
            
        Returns:
            Tuple of (results list, statistics list)
        """
        results_list = []
        stats_list = []
        
        # Configure main process GDAL settings
        os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = str(self.n_workers)
        
        # Create process pool
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all chunks to the pool
            futures = {}
            for chunk_data in chunks:
                chunk_id = chunk_data[0]
                future = executor.submit(
                    process_chunk_worker,
                    chunk_data,
                    raster_path,
                    method
                )
                futures[future] = chunk_id
            
            # Process results as they complete
            if self.enable_progress:
                progress_bar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing chunks"
                )
            else:
                progress_bar = as_completed(futures)
            
            for future in progress_bar:
                chunk_id = futures[future]
                
                try:
                    # Get result from future
                    chunk_id, results_df, stats = future.result(timeout=300)  # 5 min timeout
                    
                    if not results_df.empty:
                        results_list.append(results_df)
                        stats_list.append(stats)
                        
                        # Call checkpoint callback if provided
                        if checkpoint_callback:
                            checkpoint_callback(chunk_id, results_df)
                        
                        # Log successful completion
                        logger.debug(f"Completed chunk {chunk_id}: {stats['n_processed']} parcels")
                    else:
                        logger.warning(f"Empty results for chunk {chunk_id}")
                        stats_list.append(stats)
                        
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_id}: {str(e)}")
                    
                    # Add error statistics
                    error_stats = {
                        'chunk_id': chunk_id,
                        'status': 'error',
                        'error': str(e)
                    }
                    stats_list.append(error_stats)
                
                # Check memory usage periodically
                if len(results_list) % 10 == 0:
                    self._check_memory_usage()
        
        logger.info(f"Processed {len(results_list)} chunks successfully")
        
        return results_list, stats_list
    
    def _check_memory_usage(self):
        """Check current memory usage and log warnings if high."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 90:
            logger.error(f"Critical memory usage: {memory_percent:.1f}%")
        elif memory_percent > 80:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}%")
    
    def process_with_spatial_ordering(
        self,
        chunks: List[Tuple[int, gpd.GeoDataFrame]],
        raster_path: str,
        method: str = 'subpixel',
        checkpoint_callback: Optional[callable] = None
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """Process chunks with spatial ordering for better cache efficiency.
        
        Args:
            chunks: List of (chunk_id, chunk_gdf) tuples
            raster_path: Path to raster file
            method: Processing method
            checkpoint_callback: Optional callback for saving checkpoints
            
        Returns:
            Tuple of (results list, statistics list)
        """
        # Sort chunks by spatial proximity (simple approach using centroids)
        logger.info("Sorting chunks by spatial proximity")
        
        chunk_centroids = []
        for chunk_id, chunk_gdf in chunks:
            # Calculate centroid of chunk bounds
            bounds = chunk_gdf.total_bounds
            centroid_x = (bounds[0] + bounds[2]) / 2
            centroid_y = (bounds[1] + bounds[3]) / 2
            chunk_centroids.append((chunk_id, centroid_x, centroid_y))
        
        # Sort by x, then y coordinate (simple spatial ordering)
        chunk_centroids.sort(key=lambda x: (x[1], x[2]))
        
        # Reorder chunks based on spatial sorting
        sorted_chunks = []
        chunk_dict = {chunk_id: chunk_gdf for chunk_id, chunk_gdf in chunks}
        
        for chunk_id, _, _ in chunk_centroids:
            sorted_chunks.append((chunk_id, chunk_dict[chunk_id]))
        
        logger.info("Processing chunks in spatial order")
        
        # Process with the sorted chunks
        return self.process_chunks_parallel(
            sorted_chunks,
            raster_path,
            method,
            checkpoint_callback
        )