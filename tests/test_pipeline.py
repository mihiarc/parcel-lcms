#!/usr/bin/env python
"""Test script for the zonal statistics pipeline."""

import sys
import warnings
warnings.filterwarnings('ignore')

# Import the pipeline
from src.main import run_pipeline

if __name__ == "__main__":
    print("Testing zonal statistics pipeline with 100 sample parcels...")
    print("-" * 60)
    
    try:
        # Run with dry run first to test data loading
        print("Step 1: Testing data loading (dry run)...")
        run_pipeline(
            sample_size=100,
            dry_run=True,
            chunk_size=50
        )
        
        print("\nDry run successful!")
        print("-" * 60)
        print("\nStep 2: Processing 100 sample parcels...")
        
        # Now run actual processing on small sample
        run_pipeline(
            sample_size=100,
            chunk_size=50,
            strategy="count",
            output_format="csv"
        )
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)