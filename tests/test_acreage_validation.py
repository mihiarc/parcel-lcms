#!/usr/bin/env python
"""Test script to validate acreage calculations between CAL_ACREAGE and pixel-based measurements."""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Import pipeline components
from src.config import RASTER_PATH, PARCEL_PATH, PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.zonal_processor import ZonalStatsProcessor

def calculate_acreage_from_pixels(pixel_count, pixel_area_m2):
    """Convert pixel count to acreage.
    
    Args:
        pixel_count: Number of pixels
        pixel_area_m2: Area of each pixel in square meters
        
    Returns:
        Acreage (1 acre = 4046.86 m²)
    """
    total_area_m2 = pixel_count * pixel_area_m2
    acres = total_area_m2 / 4046.86  # Convert m² to acres
    return acres

def test_acreage_comparison(sample_size=100):
    """Test acreage comparison between CAL_ACREAGE and pixel-based calculations.
    
    Args:
        sample_size: Number of parcels to test
    """
    print("=" * 70)
    print("ACREAGE VALIDATION TEST")
    print("=" * 70)
    print(f"Testing {sample_size} parcels...")
    print()
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    processor = ZonalStatsProcessor()
    
    # Load raster metadata
    print("Loading raster metadata...")
    raster_metadata = data_loader.load_raster_metadata(RASTER_PATH)
    
    # Get pixel resolution (assuming square pixels)
    pixel_width = raster_metadata['resolution'][0]  # meters
    pixel_height = raster_metadata['resolution'][1]  # meters
    pixel_area_m2 = pixel_width * pixel_height
    
    print(f"Raster resolution: {pixel_width}m x {pixel_height}m")
    print(f"Pixel area: {pixel_area_m2:.2f} m² ({pixel_area_m2/4046.86:.6f} acres)")
    print()
    
    # Load sample parcels
    print(f"Loading {sample_size} sample parcels...")
    parcels = data_loader.load_parcels(
        PARCEL_PATH,
        columns=[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry'],
        sample_size=sample_size
    )
    
    # Transform to raster CRS
    print("Transforming parcels to raster CRS...")
    parcels_transformed = preprocessor.transform_parcels_crs(
        parcels,
        target_crs=raster_metadata['crs']
    )
    
    # Calculate area in transformed CRS
    print("Calculating geometric areas...")
    parcels_transformed['geometry_area_m2'] = parcels_transformed.geometry.area
    parcels_transformed['geometry_acres'] = parcels_transformed['geometry_area_m2'] / 4046.86
    
    # Process zonal statistics
    print("Calculating zonal statistics...")
    results = processor.calculate_land_use_proportions(
        parcels_transformed,
        str(RASTER_PATH)
    )
    
    # Merge results
    comparison_df = parcels_transformed[[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 
                                        'geometry_area_m2', 'geometry_acres']].merge(
        results[[PARCEL_ID_FIELD, 'total_pixels', 'valid_pixels']],
        on=PARCEL_ID_FIELD
    )
    
    # Calculate pixel-based acreage
    comparison_df['pixel_area_m2'] = comparison_df['total_pixels'] * pixel_area_m2
    comparison_df['pixel_acres'] = comparison_df['pixel_area_m2'] / 4046.86
    
    # Calculate differences
    comparison_df['diff_acres_cal_vs_pixel'] = (
        comparison_df[PARCEL_ACREAGE_FIELD] - comparison_df['pixel_acres']
    )
    comparison_df['diff_pct_cal_vs_pixel'] = (
        (comparison_df['diff_acres_cal_vs_pixel'] / comparison_df[PARCEL_ACREAGE_FIELD]) * 100
    )
    
    comparison_df['diff_acres_cal_vs_geom'] = (
        comparison_df[PARCEL_ACREAGE_FIELD] - comparison_df['geometry_acres']
    )
    comparison_df['diff_pct_cal_vs_geom'] = (
        (comparison_df['diff_acres_cal_vs_geom'] / comparison_df[PARCEL_ACREAGE_FIELD]) * 100
    )
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("ACREAGE COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n1. OVERALL STATISTICS")
    print("-" * 40)
    print(f"Total parcels analyzed: {len(comparison_df)}")
    print(f"Parcels with valid pixels: {(comparison_df['valid_pixels'] > 0).sum()}")
    print(f"Parcels with no pixels: {(comparison_df['valid_pixels'] == 0).sum()}")
    
    print("\n2. ACREAGE SUMMARY (acres)")
    print("-" * 40)
    stats_cols = [PARCEL_ACREAGE_FIELD, 'geometry_acres', 'pixel_acres']
    summary_stats = comparison_df[stats_cols].describe()
    print(summary_stats)
    
    print("\n3. DIFFERENCE ANALYSIS: CAL_ACREAGE vs PIXEL-BASED")
    print("-" * 40)
    
    # Filter out parcels with no pixels for meaningful comparison
    valid_comparison = comparison_df[comparison_df['valid_pixels'] > 0].copy()
    
    if len(valid_comparison) > 0:
        print(f"Parcels with valid pixels: {len(valid_comparison)}")
        print(f"\nAbsolute difference (acres):")
        print(f"  Mean:   {valid_comparison['diff_acres_cal_vs_pixel'].mean():+.3f}")
        print(f"  Median: {valid_comparison['diff_acres_cal_vs_pixel'].median():+.3f}")
        print(f"  Std:    {valid_comparison['diff_acres_cal_vs_pixel'].std():.3f}")
        print(f"  Min:    {valid_comparison['diff_acres_cal_vs_pixel'].min():+.3f}")
        print(f"  Max:    {valid_comparison['diff_acres_cal_vs_pixel'].max():+.3f}")
        
        print(f"\nPercentage difference (%):")
        print(f"  Mean:   {valid_comparison['diff_pct_cal_vs_pixel'].mean():+.1f}%")
        print(f"  Median: {valid_comparison['diff_pct_cal_vs_pixel'].median():+.1f}%")
        print(f"  Std:    {valid_comparison['diff_pct_cal_vs_pixel'].std():.1f}%")
        
        # Categorize differences
        print(f"\nDifference categories:")
        print(f"  Within ±5%:   {(valid_comparison['diff_pct_cal_vs_pixel'].abs() <= 5).sum()} parcels")
        print(f"  Within ±10%:  {(valid_comparison['diff_pct_cal_vs_pixel'].abs() <= 10).sum()} parcels")
        print(f"  Within ±25%:  {(valid_comparison['diff_pct_cal_vs_pixel'].abs() <= 25).sum()} parcels")
        print(f"  Beyond ±25%:  {(valid_comparison['diff_pct_cal_vs_pixel'].abs() > 25).sum()} parcels")
    
    print("\n4. DIFFERENCE ANALYSIS: CAL_ACREAGE vs GEOMETRY-BASED")
    print("-" * 40)
    print(f"\nAbsolute difference (acres):")
    print(f"  Mean:   {comparison_df['diff_acres_cal_vs_geom'].mean():+.3f}")
    print(f"  Median: {comparison_df['diff_acres_cal_vs_geom'].median():+.3f}")
    print(f"  Std:    {comparison_df['diff_acres_cal_vs_geom'].std():.3f}")
    
    print(f"\nPercentage difference (%):")
    print(f"  Mean:   {comparison_df['diff_pct_cal_vs_geom'].mean():+.1f}%")
    print(f"  Median: {comparison_df['diff_pct_cal_vs_geom'].median():+.1f}%")
    
    # Show examples of large discrepancies
    print("\n5. LARGEST DISCREPANCIES (CAL_ACREAGE vs PIXEL-BASED)")
    print("-" * 40)
    
    if len(valid_comparison) > 0:
        # Sort by absolute percentage difference
        valid_comparison['abs_diff_pct'] = valid_comparison['diff_pct_cal_vs_pixel'].abs()
        largest_diff = valid_comparison.nlargest(min(5, len(valid_comparison)), 'abs_diff_pct')
        
        for idx, row in largest_diff.iterrows():
            print(f"\nParcel {row[PARCEL_ID_FIELD]}:")
            print(f"  CAL_ACREAGE:  {row[PARCEL_ACREAGE_FIELD]:.2f} acres")
            print(f"  Pixel-based:  {row['pixel_acres']:.2f} acres")
            print(f"  Geometry:     {row['geometry_acres']:.2f} acres")
            print(f"  Pixel count:  {row['total_pixels']}")
            print(f"  Difference:   {row['diff_acres_cal_vs_pixel']:+.2f} acres ({row['diff_pct_cal_vs_pixel']:+.1f}%)")
    
    # Save detailed results
    output_file = Path("outputs") / "acreage_comparison_results.csv"
    output_file.parent.mkdir(exist_ok=True)
    comparison_df.to_csv(output_file, index=False)
    print(f"\n6. DETAILED RESULTS")
    print("-" * 40)
    print(f"Detailed results saved to: {output_file}")
    
    # Create scatter plots data
    print("\n7. CORRELATION ANALYSIS")
    print("-" * 40)
    
    if len(valid_comparison) > 0:
        # Calculate correlations
        corr_cal_pixel = valid_comparison[PARCEL_ACREAGE_FIELD].corr(valid_comparison['pixel_acres'])
        corr_cal_geom = comparison_df[PARCEL_ACREAGE_FIELD].corr(comparison_df['geometry_acres'])
        
        print(f"Correlation CAL_ACREAGE vs Pixel-based: {corr_cal_pixel:.4f}")
        print(f"Correlation CAL_ACREAGE vs Geometry:    {corr_cal_geom:.4f}")
        
        # Calculate RMSE
        rmse_pixel = np.sqrt(((valid_comparison[PARCEL_ACREAGE_FIELD] - valid_comparison['pixel_acres'])**2).mean())
        rmse_geom = np.sqrt(((comparison_df[PARCEL_ACREAGE_FIELD] - comparison_df['geometry_acres'])**2).mean())
        
        print(f"\nRoot Mean Square Error (RMSE):")
        print(f"  CAL_ACREAGE vs Pixel-based: {rmse_pixel:.3f} acres")
        print(f"  CAL_ACREAGE vs Geometry:    {rmse_geom:.3f} acres")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return comparison_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test acreage calculations")
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of parcels to test (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        results = test_acreage_comparison(sample_size=args.sample)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)