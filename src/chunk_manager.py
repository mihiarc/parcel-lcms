"""Chunk management for memory-efficient processing."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import json
import pickle

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box

from .config import (
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    TEMP_DIR,
    CHECKPOINT_INTERVAL
)

logger = logging.getLogger(__name__)

class ChunkManager:
    """Manage spatial chunking and checkpoint/resume functionality."""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        checkpoint_dir: Optional[Path] = None
    ):
        """Initialize chunk manager.
        
        Args:
            chunk_size: Target number of parcels per chunk
            checkpoint_dir: Directory for checkpoint files
        """
        self.chunk_size = chunk_size
        self.checkpoint_dir = checkpoint_dir or TEMP_DIR
        self.checkpoint_file = self.checkpoint_dir / "processing_checkpoint.json"
        self.chunks = []
        self.processed_chunks = set()
        
    def create_spatial_chunks(
        self,
        parcels: gpd.GeoDataFrame,
        strategy: str = "count"
    ) -> List[gpd.GeoDataFrame]:
        """Create spatial chunks for processing.
        
        Args:
            parcels: Input parcels
            strategy: Chunking strategy ('count', 'spatial', 'hybrid')
            
        Returns:
            List of parcel chunks
        """
        logger.info(f"Creating chunks using {strategy} strategy")
        
        if strategy == "count":
            chunks = self._chunk_by_count(parcels)
        elif strategy == "spatial":
            chunks = self._chunk_by_spatial_grid(parcels)
        elif strategy == "hybrid":
            chunks = self._chunk_hybrid(parcels)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks")
        
        # Log chunk statistics
        chunk_sizes = [len(chunk) for chunk in chunks]
        logger.info(f"Chunk sizes: min={min(chunk_sizes)}, "
                   f"max={max(chunk_sizes)}, "
                   f"mean={np.mean(chunk_sizes):.1f}")
        
        return chunks
    
    def _chunk_by_count(self, parcels: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
        """Simple count-based chunking.
        
        Args:
            parcels: Input parcels
            
        Returns:
            List of chunks
        """
        chunks = []
        n_parcels = len(parcels)
        
        for i in range(0, n_parcels, self.chunk_size):
            chunk = parcels.iloc[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_spatial_grid(self, parcels: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
        """Spatial grid-based chunking.
        
        Args:
            parcels: Input parcels
            
        Returns:
            List of chunks
        """
        # Get bounds
        minx, miny, maxx, maxy = parcels.total_bounds
        
        # Calculate grid dimensions
        n_cells = max(1, int(np.sqrt(len(parcels) / self.chunk_size)))
        x_step = (maxx - minx) / n_cells
        y_step = (maxy - miny) / n_cells
        
        logger.info(f"Creating {n_cells}x{n_cells} spatial grid")
        
        chunks = []
        
        # Create grid cells and assign parcels
        for i in range(n_cells):
            for j in range(n_cells):
                # Create cell bounds
                cell_minx = minx + i * x_step
                cell_maxx = minx + (i + 1) * x_step
                cell_miny = miny + j * y_step
                cell_maxy = miny + (j + 1) * y_step
                
                # Create cell geometry
                cell = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
                
                # Find parcels in this cell
                mask = parcels.geometry.intersects(cell)
                chunk = parcels[mask]
                
                if len(chunk) > 0:
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_hybrid(self, parcels: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
        """Hybrid chunking combining spatial and count strategies.
        
        Args:
            parcels: Input parcels
            
        Returns:
            List of chunks
        """
        # Start with spatial grid
        spatial_chunks = self._chunk_by_spatial_grid(parcels)
        
        # Refine chunks that are too large or too small
        refined_chunks = []
        
        for chunk in spatial_chunks:
            chunk_size = len(chunk)
            
            if chunk_size > MAX_CHUNK_SIZE:
                # Split large chunks
                sub_chunks = self._chunk_by_count(chunk)
                refined_chunks.extend(sub_chunks)
            elif chunk_size < MIN_CHUNK_SIZE and refined_chunks:
                # Merge small chunks with previous
                refined_chunks[-1] = pd.concat([refined_chunks[-1], chunk])
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks
    
    def save_checkpoint(self, chunk_id: int, results: pd.DataFrame) -> None:
        """Save processing checkpoint.
        
        Args:
            chunk_id: Completed chunk ID
            results: Processing results
        """
        self.processed_chunks.add(chunk_id)
        
        checkpoint_data = {
            'processed_chunks': list(self.processed_chunks),
            'total_chunks': len(self.chunks),
            'last_chunk_id': chunk_id
        }
        
        # Save checkpoint
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Save results
        results_file = self.checkpoint_dir / f"chunk_{chunk_id}_results.parquet"
        results.to_parquet(results_file)
        
        logger.debug(f"Saved checkpoint for chunk {chunk_id}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists.
        
        Returns:
            Checkpoint data or None
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.processed_chunks = set(checkpoint_data['processed_chunks'])
            
            logger.info(f"Loaded checkpoint: {len(self.processed_chunks)}/{checkpoint_data['total_chunks']} chunks processed")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def get_unprocessed_chunks(self) -> List[int]:
        """Get list of unprocessed chunk IDs.
        
        Returns:
            List of chunk IDs that haven't been processed
        """
        all_chunk_ids = set(range(len(self.chunks)))
        unprocessed = all_chunk_ids - self.processed_chunks
        return sorted(list(unprocessed))
    
    def iterate_chunks(
        self,
        parcels: gpd.GeoDataFrame,
        resume: bool = False
    ) -> Iterator[Tuple[int, gpd.GeoDataFrame]]:
        """Iterate over chunks with optional resume capability.
        
        Args:
            parcels: Input parcels
            resume: Whether to resume from checkpoint
            
        Yields:
            Tuple of (chunk_id, chunk_gdf)
        """
        # Create chunks if not already done
        if not self.chunks:
            self.create_spatial_chunks(parcels)
        
        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from chunk {checkpoint['last_chunk_id'] + 1}")
        
        # Get chunks to process
        chunks_to_process = self.get_unprocessed_chunks()
        
        logger.info(f"Processing {len(chunks_to_process)} chunks")
        
        for chunk_id in chunks_to_process:
            yield chunk_id, self.chunks[chunk_id]
    
    def merge_chunk_results(self) -> pd.DataFrame:
        """Merge results from all processed chunks.
        
        Returns:
            Merged results DataFrame
        """
        logger.info("Merging chunk results")
        
        results_files = sorted(self.checkpoint_dir.glob("chunk_*_results.parquet"))
        
        if not results_files:
            logger.warning("No chunk results found")
            return pd.DataFrame()
        
        # Load and merge results
        results_list = []
        for file in results_files:
            chunk_results = pd.read_parquet(file)
            results_list.append(chunk_results)
        
        merged_results = pd.concat(results_list, ignore_index=True)
        
        logger.info(f"Merged {len(results_list)} chunks with {len(merged_results)} total results")
        
        return merged_results
    
    def cleanup_checkpoints(self) -> None:
        """Clean up checkpoint files."""
        logger.info("Cleaning up checkpoint files")
        
        # Remove checkpoint file
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        # Remove chunk result files
        for file in self.checkpoint_dir.glob("chunk_*_results.parquet"):
            file.unlink()
        
        logger.info("Checkpoint cleanup complete")