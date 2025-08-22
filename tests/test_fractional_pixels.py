#!/usr/bin/env python
"""Test fractional pixel counting vs whole pixel counting."""

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
from src.zonal_processor_fractional import FractionalZonalStatsProcessor

def compare_methods(sample_size=50):
    """Compare whole pixel vs fractional pixel methods.
    
    Args:
        sample_size: Number of parcels to test
    """
    print("=" * 80)
    print("FRACTIONAL vs WHOLE PIXEL COMPARISON TEST")
    print("=" * 80)
    print(f"Testing {sample_size} parcels with both methods...")
    print()
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    whole_processor = ZonalStatsProcessor()
    fractional_processor = FractionalZonalStatsProcessor()
    
    # Load raster metadata
    print("Loading raster metadata...")
    raster_metadata = data_loader.load_raster_metadata(RASTER_PATH)
    
    # Get pixel resolution
    pixel_width = raster_metadata['resolution'][0]
    pixel_height = raster_metadata['resolution'][1]
    pixel_area_m2 = pixel_width * pixel_height
    pixel_area_acres = pixel_area_m2 / 4046.86
    
    print(f"Pixel resolution: {pixel_width}m x {pixel_height}m")
    print(f"Pixel area: {pixel_area_m2:.2f} m² ({pixel_area_acres:.6f} acres)")
    print()
    
    # Load sample parcels - focus on small parcels
    print(f"Loading {sample_size} sample parcels (focusing on small parcels)...")
    all_parcels = data_loader.load_parcels(
        PARCEL_PATH,
        columns=[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry']
    )
    
    # Sort by acreage and take smallest parcels for better demonstration
    all_parcels_sorted = all_parcels.sort_values(PARCEL_ACREAGE_FIELD)
    parcels = all_parcels_sorted.head(sample_size)
    
    print(f"Selected parcels range: {parcels[PARCEL_ACREAGE_FIELD].min():.3f} - {parcels[PARCEL_ACREAGE_FIELD].max():.3f} acres")
    
    # Transform to raster CRS
    print("\nTransforming parcels to raster CRS...")
    parcels_transformed = preprocessor.transform_parcels_crs(
        parcels,
        target_crs=raster_metadata['crs']
    )
    
    # Calculate geometric area
    parcels_transformed['geometry_area_m2'] = parcels_transformed.geometry.area
    parcels_transformed['geometry_acres'] = parcels_transformed['geometry_area_m2'] / 4046.86
    
    # Process with WHOLE pixel method
    print("\n1. Processing with WHOLE PIXEL method...")
    whole_results = whole_processor.calculate_land_use_proportions(
        parcels_transformed,
        str(RASTER_PATH)
    )
    
    # Process with FRACTIONAL pixel method
    print("\n2. Processing with FRACTIONAL PIXEL method...")
    fractional_results = fractional_processor.calculate_land_use_proportions_fractional(
        parcels_transformed,
        str(RASTER_PATH),
        use_fractional=True
    )
    
    # Merge results for comparison
    comparison = parcels_transformed[[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry_acres']].merge(
        whole_results[[PARCEL_ID_FIELD, 'total_pixels', 'valid_pixels']].rename(
            columns={'total_pixels': 'whole_pixels', 'valid_pixels': 'whole_valid_pixels'}
        ),
        on=PARCEL_ID_FIELD
    ).merge(
        fractional_results[[PARCEL_ID_FIELD, 'total_pixels', 'fractional_acres', 
                          'whole_pixels_covered', 'acre_diff_pct']].rename(
            columns={'total_pixels': 'fractional_pixels'}
        ),
        on=PARCEL_ID_FIELD
    )
    
    # Calculate whole pixel acres
    comparison['whole_pixel_acres'] = comparison['whole_pixels'] * pixel_area_acres
    
    # Calculate errors
    comparison['whole_error_acres'] = comparison['whole_pixel_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    comparison['whole_error_pct'] = (comparison['whole_error_acres'] / comparison[PARCEL_ACREAGE_FIELD]) * 100
    
    comparison['frac_error_acres'] = comparison['fractional_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    comparison['frac_error_pct'] = (comparison['frac_error_acres'] / comparison[PARCEL_ACREAGE_FIELD]) * 100
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print("\n1. SUMMARY STATISTICS")
    print("-" * 50)
    
    print(f"\nWHOLE PIXEL METHOD:")
    print(f"  Mean absolute error: {comparison['whole_error_acres'].abs().mean():.3f} acres")
    print(f"  Mean percentage error: {comparison['whole_error_pct'].mean():.1f}%")
    print(f"  Median percentage error: {comparison['whole_error_pct'].median():.1f}%")
    print(f"  RMSE: {np.sqrt((comparison['whole_error_acres']**2).mean()):.3f} acres")
    
    print(f"\nFRACTIONAL PIXEL METHOD:")
    print(f"  Mean absolute error: {comparison['frac_error_acres'].abs().mean():.3f} acres")
    print(f"  Mean percentage error: {comparison['frac_error_pct'].mean():.1f}%")
    print(f"  Median percentage error: {comparison['frac_error_pct'].median():.1f}%")
    print(f"  RMSE: {np.sqrt((comparison['frac_error_acres']**2).mean()):.3f} acres")
    
    print("\n2. ERROR REDUCTION")
    print("-" * 50)
    
    # Calculate improvement
    whole_abs_error = comparison['whole_error_acres'].abs()
    frac_abs_error = comparison['frac_error_acres'].abs()
    error_reduction = whole_abs_error - frac_abs_error
    
    print(f"Mean error reduction: {error_reduction.mean():.3f} acres")
    print(f"Parcels with improved accuracy: {(error_reduction > 0).sum()} / {len(comparison)}")
    print(f"Average improvement for small parcels (<1 acre): {error_reduction[comparison[PARCEL_ACREAGE_FIELD] < 1].mean():.3f} acres")
    
    print("\n3. ACCURACY BY PARCEL SIZE")
    print("-" * 50)
    
    # Group by size categories
    size_bins = [0, 0.5, 1, 2, 5, float('inf')]
    size_labels = ['<0.5 acres', '0.5-1 acres', '1-2 acres', '2-5 acres', '>5 acres']
    comparison['size_category'] = pd.cut(comparison[PARCEL_ACREAGE_FIELD], bins=size_bins, labels=size_labels)
    
    for category in size_labels:
        cat_data = comparison[comparison['size_category'] == category]
        if len(cat_data) > 0:
            print(f"\n{category} ({len(cat_data)} parcels):")
            print(f"  Whole pixel mean error: {cat_data['whole_error_pct'].mean():+.1f}%")
            print(f"  Fractional mean error:  {cat_data['frac_error_pct'].mean():+.1f}%")
    
    print("\n4. EXAMPLE PARCELS (Smallest 10)")
    print("-" * 50)
    
    examples = comparison.nsmallest(10, PARCEL_ACREAGE_FIELD)
    
    for idx, row in examples.iterrows():
        print(f"\nParcel {row[PARCEL_ID_FIELD]}:")
        print(f"  Actual acreage:     {row[PARCEL_ACREAGE_FIELD]:.4f} acres")
        print(f"  Whole pixels:       {row['whole_pixels']:.0f} pixels = {row['whole_pixel_acres']:.4f} acres (error: {row['whole_error_pct']:+.1f}%)")
        print(f"  Fractional pixels:  {row['fractional_pixels']:.2f} pixels = {row['fractional_acres']:.4f} acres (error: {row['frac_error_pct']:+.1f}%)")
        print(f"  Error reduction:    {abs(row['whole_error_acres']) - abs(row['frac_error_acres']):.4f} acres")
    
    print("\n5. PIXEL COVERAGE ANALYSIS")
    print("-" * 50)
    
    print(f"\nPixel counts:")
    print(f"  Parcels with <1 whole pixel: {(comparison['whole_pixels'] < 1).sum()}")
    print(f"  Parcels with <1 fractional pixel: {(comparison['fractional_pixels'] < 1).sum()}")
    print(f"  Mean whole pixels per parcel: {comparison['whole_pixels'].mean():.1f}")
    print(f"  Mean fractional pixels per parcel: {comparison['fractional_pixels'].mean():.2f}")
    
    # Save detailed results
    output_file = Path("outputs") / "fractional_pixel_comparison.csv"
    output_file.parent.mkdir(exist_ok=True)
    comparison.to_csv(output_file, index=False)
    
    print("\n6. OUTPUT")
    print("-" * 50)
    print(f"Detailed results saved to: {output_file}")
    
    # Create visualization data
    print("\n7. CORRELATION ANALYSIS")
    print("-" * 50)
    
    corr_whole = comparison[PARCEL_ACREAGE_FIELD].corr(comparison['whole_pixel_acres'])
    corr_frac = comparison[PARCEL_ACREAGE_FIELD].corr(comparison['fractional_acres'])
    
    print(f"Correlation with CAL_ACREAGE:")
    print(f"  Whole pixel method:      {corr_whole:.6f}")
    print(f"  Fractional pixel method: {corr_frac:.6f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if frac_abs_error.mean() < whole_abs_error.mean():
        improvement_pct = ((whole_abs_error.mean() - frac_abs_error.mean()) / whole_abs_error.mean()) * 100
        print(f"✓ Fractional pixel method reduces mean absolute error by {improvement_pct:.1f}%")
        print(f"✓ Particularly effective for parcels smaller than {2 * pixel_area_acres:.2f} acres (2 pixels)")
    else:
        print("✗ Fractional pixel method did not improve accuracy overall")
    
    print("\n" + "=" * 80)
    
    return comparison

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare fractional vs whole pixel methods")
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of parcels to test (default: 50)"
    )
    
    args = parser.parse_args()
    
    try:
        results = compare_methods(sample_size=args.sample)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)