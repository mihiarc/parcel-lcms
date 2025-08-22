"""Main script to run the parcel land use zonal statistics pipeline."""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import pandas as pd

from .config import (
    setup_logging,
    RASTER_PATH,
    PARCEL_PATH,
    RASTER_CRS,
    CHUNK_SIZE,
    N_WORKERS,
    ENABLE_PROGRESS_BAR,
    PARALLEL_PROCESSING,
    SPATIAL_ORDERING
)
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor  
from .zonal_processor import ZonalStatsProcessor
from .chunk_manager import ChunkManager
from .result_aggregator import ResultAggregator
from .parallel_processor import ParallelChunkProcessor

logger = logging.getLogger(__name__)

def run_pipeline(
    raster_path: Path = RASTER_PATH,
    parcel_path: Path = PARCEL_PATH,
    chunk_size: int = CHUNK_SIZE,
    strategy: str = "hybrid",
    sample_size: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
    output_format: str = "geoparquet",
    method: str = "subpixel",
    parallel: Optional[bool] = None
) -> None:
    """Run the complete zonal statistics pipeline.
    
    Args:
        raster_path: Path to land use raster
        parcel_path: Path to parcel data
        chunk_size: Number of parcels per chunk
        strategy: Chunking strategy
        sample_size: Sample size for testing
        resume: Resume from checkpoint
        dry_run: Perform dry run without processing
        output_format: Output file format
        method: Zonal stats method ('subpixel', 'standard', 'center')
        parallel: Use parallel processing (default: from config)
    """
    logger.info("=" * 60)
    logger.info("PARCEL LAND USE ZONAL STATISTICS PIPELINE")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize components
        logger.info("Initializing pipeline components")
        logger.info(f"Using {method} method for zonal statistics")
        
        # Determine if using parallel processing
        use_parallel = parallel if parallel is not None else PARALLEL_PROCESSING
        if use_parallel:
            logger.info(f"Parallel processing enabled with {N_WORKERS} workers")
        else:
            logger.info("Sequential processing mode")
        
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        processor = ZonalStatsProcessor(n_workers=N_WORKERS)
        chunk_manager = ChunkManager(chunk_size=chunk_size)
        aggregator = ResultAggregator()
        
        # Step 1: Load data
        logger.info("-" * 40)
        logger.info("STEP 1: Loading data")
        
        # Load raster metadata
        raster_metadata = data_loader.load_raster_metadata(raster_path)
        
        # Load parcels
        parcels = data_loader.load_parcels(
            parcel_path,
            sample_size=sample_size
        )
        
        # Validate compatibility
        data_loader.validate_data_compatibility()
        
        # Estimate requirements
        estimates = data_loader.estimate_processing_requirements()
        
        if dry_run:
            logger.info("DRY RUN MODE - Analysis complete")
            logger.info(f"Would process {len(parcels)} parcels")
            logger.info(f"Estimated chunks: {estimates.get('estimated_chunks', 'N/A')}")
            logger.info(f"Estimated memory: {estimates.get('estimated_memory_peak_mb', 'N/A')} MB")
            return
        
        # Step 2: Preprocess parcels
        logger.info("-" * 40)
        logger.info("STEP 2: Preprocessing parcels")
        
        parcels_processed = preprocessor.prepare_for_processing(
            parcels,
            target_crs=RASTER_CRS,
            raster_bounds=raster_metadata['bounds'],
            min_area_m2=0  # Process all parcels, no minimum size
        )
        
        # Save original parcels for geometry join later
        parcels_original = parcels_processed.copy()
        
        # Step 3: Process chunks
        logger.info("-" * 40)
        logger.info("STEP 3: Processing parcels in chunks")
        
        results_list = []
        chunk_stats = []
        
        if use_parallel:
            # Parallel processing mode
            logger.info("Using parallel processing")
            
            # Collect all chunks
            all_chunks = []
            for chunk_id, chunk in chunk_manager.iterate_chunks(parcels_processed, resume=resume):
                all_chunks.append((chunk_id, chunk))
            
            # Initialize parallel processor
            parallel_processor = ParallelChunkProcessor(
                n_workers=N_WORKERS,
                enable_progress=ENABLE_PROGRESS_BAR
            )
            
            # Define checkpoint callback
            def checkpoint_callback(chunk_id, results_df):
                chunk_manager.save_checkpoint(chunk_id, results_df)
            
            # Process chunks in parallel
            if SPATIAL_ORDERING:
                results_list, chunk_stats = parallel_processor.process_with_spatial_ordering(
                    all_chunks,
                    str(raster_path),
                    method=method,
                    checkpoint_callback=checkpoint_callback
                )
            else:
                results_list, chunk_stats = parallel_processor.process_chunks_parallel(
                    all_chunks,
                    str(raster_path),
                    method=method,
                    checkpoint_callback=checkpoint_callback
                )
        else:
            # Sequential processing mode (original implementation)
            logger.info("Using sequential processing")
            
            # Create progress bar if enabled
            if ENABLE_PROGRESS_BAR:
                chunk_iterator = tqdm(
                    chunk_manager.iterate_chunks(parcels_processed, resume=resume),
                    total=len(chunk_manager.get_unprocessed_chunks()) if resume else len(parcels_processed) // chunk_size + 1,
                    desc="Processing chunks"
                )
            else:
                chunk_iterator = chunk_manager.iterate_chunks(parcels_processed, resume=resume)
            
            # Process each chunk
            for chunk_id, chunk in chunk_iterator:
                logger.info(f"Processing chunk {chunk_id} ({len(chunk)} parcels)")
                
                # Calculate zonal statistics with specified method
                chunk_results, stats = processor.process_chunk(
                    chunk,
                    str(raster_path),
                    chunk_id,
                    method=method
                )
                
                if not chunk_results.empty:
                    results_list.append(chunk_results)
                    chunk_stats.append(stats)
                    
                    # Save checkpoint
                    chunk_manager.save_checkpoint(chunk_id, chunk_results)
                
                # Log progress
                if chunk_id % 10 == 0:
                    processed = sum(len(r) for r in results_list)
                    logger.info(f"Progress: {processed}/{len(parcels_processed)} parcels processed")
        
        # Step 4: Aggregate results
        logger.info("-" * 40)
        logger.info("STEP 4: Aggregating results")
        
        if resume and not results_list:
            # Load from checkpoints
            results_df = chunk_manager.merge_chunk_results()
            results_gdf = aggregator.aggregate_results([results_df], parcels_original)
        else:
            results_gdf = aggregator.aggregate_results(results_list, parcels_original)
        
        # Validate results
        results_gdf = processor.validate_results(results_gdf)
        
        # Step 5: Save results
        logger.info("-" * 40)
        logger.info("STEP 5: Saving results")
        
        # Save main results
        output_file = aggregator.save_results(results_gdf, format=output_format)
        
        # Generate and save summary
        summary = aggregator.generate_summary_statistics(results_gdf)
        processing_stats = processor.get_processing_summary()
        summary['processing_stats'] = processing_stats
        
        aggregator.save_summary(summary)
        
        # Create report
        report = aggregator.create_report(results_gdf, processing_stats)
        print("\n" + report)
        
        # Clean up checkpoints
        if not resume:
            chunk_manager.cleanup_checkpoints()
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info("-" * 40)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Output saved to: {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Calculate land use proportions within parcels using zonal statistics"
    )
    
    parser.add_argument(
        "--raster",
        type=Path,
        default=RASTER_PATH,
        help="Path to land use raster file"
    )
    
    parser.add_argument(
        "--parcels",
        type=Path,
        default=PARCEL_PATH,
        help="Path to parcel boundaries file"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Number of parcels per processing chunk"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["count", "spatial", "hybrid"],
        default="hybrid",
        help="Chunking strategy to use"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample size for testing (processes only N parcels)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without processing"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["geoparquet", "geojson", "shapefile", "csv"],
        default="geoparquet",
        help="Output file format"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--method",
        choices=["subpixel", "standard", "center"],
        default="subpixel",
        help="Zonal statistics method (default: subpixel for 99%% better accuracy)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (uses multiple CPU cores)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (force sequential mode)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    setup_logging()
    
    # Determine parallel processing setting
    parallel = None
    if args.parallel:
        parallel = True
    elif args.no_parallel:
        parallel = False
    # Otherwise, use default from config
    
    # Run pipeline
    run_pipeline(
        raster_path=args.raster,
        parcel_path=args.parcels,
        chunk_size=args.chunk_size,
        strategy=args.strategy,
        sample_size=args.sample,
        resume=args.resume,
        dry_run=args.dry_run,
        output_format=args.output_format,
        method=args.method,
        parallel=parallel
    )

if __name__ == "__main__":
    main()