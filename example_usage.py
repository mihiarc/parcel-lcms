#!/usr/bin/env python3
"""
Example usage of the zonal statistics pipeline.

This script demonstrates how to use the pipeline for processing
parcel data against land use rasters.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.main import ZonalStatsPipeline
from src.config import get_config

def main():
    """Run example pipeline."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting zonal statistics pipeline example")
    
    try:
        # Get default configuration
        config = get_config()
        
        # Optionally modify configuration for testing
        # config['parcels_per_chunk'] = 1000  # Smaller chunks for testing
        
        # Create pipeline
        pipeline = ZonalStatsPipeline(config)
        
        # Run pipeline with hybrid chunking strategy
        results = pipeline.run_pipeline(
            resume_from_checkpoint=True,
            chunking_strategy='hybrid'
        )
        
        # Print results
        if results['success']:
            stats = results['pipeline_stats']
            print(f"\n=== PROCESSING COMPLETED ===")
            print(f"Total parcels: {stats['total_parcels']:,}")
            print(f"Processed: {stats['processed_parcels']:,}")
            print(f"Failed: {stats['failed_parcels']:,}")
            print(f"Success rate: {stats.get('success_rate', 0) * 100:.1f}%")
            print(f"Processing time: {stats['processing_time'] / 3600:.2f} hours")
            print(f"Peak memory: {stats['memory_peak_mb']:.0f} MB")
            
            # Output file information
            if 'processing_summary' in results:
                output_info = results['processing_summary']['processing_summary']
                print(f"Output file: {output_info['output_file']}")
                print(f"Output size: {output_info['file_size_mb']:.1f} MB")
            
            print("===============================")
        else:
            print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())