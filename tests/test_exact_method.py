#!/usr/bin/env python
"""Test exact/weighted pixel counting method for improved accuracy."""

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
from src.zonal_processor_exact import ExactZonalStatsProcessor

def test_exact_method(sample_size=50):
    """Test exact/weighted method vs standard method.
    
    Args:
        sample_size: Number of parcels to test
    """
    print("=" * 80)
    print("EXACT/WEIGHTED PIXEL METHOD TEST")
    print("=" * 80)
    print(f"Testing {sample_size} parcels...")
    print()
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    standard_processor = ZonalStatsProcessor()
    exact_processor = ExactZonalStatsProcessor()
    
    # Load raster metadata
    print("Loading raster metadata...")
    raster_metadata = data_loader.load_raster_metadata(RASTER_PATH)
    
    pixel_area_m2 = 900.0  # 30m x 30m
    pixel_area_acres = pixel_area_m2 / 4046.86
    
    print(f"Pixel area: {pixel_area_m2:.0f} m² ({pixel_area_acres:.4f} acres)")
    print()
    
    # Load parcels - focus on small ones
    print("Loading small parcels for testing...")
    all_parcels = data_loader.load_parcels(
        PARCEL_PATH,
        columns=[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry']
    )
    
    # Get smallest parcels
    parcels = all_parcels.nsmallest(sample_size, PARCEL_ACREAGE_FIELD)
    
    print(f"Parcel size range: {parcels[PARCEL_ACREAGE_FIELD].min():.4f} - "
          f"{parcels[PARCEL_ACREAGE_FIELD].max():.4f} acres")
    print(f"Parcels smaller than 1 pixel ({pixel_area_acres:.4f} acres): "
          f"{(parcels[PARCEL_ACREAGE_FIELD] < pixel_area_acres).sum()}")
    
    # Transform to raster CRS
    print("\nTransforming to raster CRS...")
    parcels_transformed = preprocessor.transform_parcels_crs(
        parcels,
        target_crs=raster_metadata['crs']
    )
    
    # Process with standard method
    print("\nProcessing with STANDARD method (all_touched=True)...")
    standard_results = standard_processor.calculate_land_use_proportions(
        parcels_transformed,
        str(RASTER_PATH)
    )
    
    # Process with exact/weighted method
    print("\nProcessing with EXACT/WEIGHTED method...")
    exact_results = exact_processor.calculate_exact_proportions(
        parcels_transformed,
        str(RASTER_PATH)
    )
    
    # Merge results
    comparison = parcels[[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD]].merge(
        standard_results[[PARCEL_ID_FIELD, 'total_pixels']].rename(
            columns={'total_pixels': 'standard_pixels'}
        ),
        on=PARCEL_ID_FIELD
    ).merge(
        exact_results[[PARCEL_ID_FIELD, 'total_pixels', 'exact_acres', 
                      'whole_pixels', 'center_pixels', 'weight_center']],
        on=PARCEL_ID_FIELD
    )
    
    # Calculate acres and errors
    comparison['standard_acres'] = comparison['standard_pixels'] * pixel_area_acres
    comparison['standard_error'] = comparison['standard_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    comparison['standard_error_pct'] = (comparison['standard_error'] / 
                                       comparison[PARCEL_ACREAGE_FIELD] * 100)
    
    comparison['exact_error'] = comparison['exact_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    comparison['exact_error_pct'] = (comparison['exact_error'] / 
                                    comparison[PARCEL_ACREAGE_FIELD] * 100)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n1. METHOD COMPARISON")
    print("-" * 50)
    
    print("\nSTANDARD METHOD (all_touched=True):")
    print(f"  Mean absolute error: {comparison['standard_error'].abs().mean():.4f} acres")
    print(f"  Median absolute error: {comparison['standard_error'].abs().median():.4f} acres")
    print(f"  Mean % error: {comparison['standard_error_pct'].mean():.1f}%")
    print(f"  RMSE: {np.sqrt((comparison['standard_error']**2).mean()):.4f} acres")
    
    print("\nEXACT/WEIGHTED METHOD:")
    print(f"  Mean absolute error: {comparison['exact_error'].abs().mean():.4f} acres")
    print(f"  Median absolute error: {comparison['exact_error'].abs().median():.4f} acres")
    print(f"  Mean % error: {comparison['exact_error_pct'].mean():.1f}%")
    print(f"  RMSE: {np.sqrt((comparison['exact_error']**2).mean()):.4f} acres")
    
    print("\n2. IMPROVEMENT ANALYSIS")
    print("-" * 50)
    
    improvement = comparison['standard_error'].abs() - comparison['exact_error'].abs()
    print(f"Mean error reduction: {improvement.mean():.4f} acres")
    print(f"Parcels with improved accuracy: {(improvement > 0).sum()} / {len(comparison)}")
    print(f"Parcels with worse accuracy: {(improvement < 0).sum()} / {len(comparison)}")
    print(f"Parcels unchanged: {(improvement == 0).sum()} / {len(comparison)}")
    
    print("\n3. ACCURACY BY SIZE")
    print("-" * 50)
    
    # Group by pixel count
    for threshold in [0.5, 1.0, 2.0]:
        threshold_acres = threshold * pixel_area_acres
        mask = comparison[PARCEL_ACREAGE_FIELD] < threshold_acres
        subset = comparison[mask]
        
        if len(subset) > 0:
            print(f"\nParcels < {threshold} pixels ({threshold_acres:.4f} acres): {len(subset)} parcels")
            print(f"  Standard mean error: {subset['standard_error_pct'].mean():+.1f}%")
            print(f"  Exact mean error:    {subset['exact_error_pct'].mean():+.1f}%")
    
    print("\n4. WEIGHTING ANALYSIS")
    print("-" * 50)
    
    print("\nWeight distribution (weight on center-only method):")
    print(f"  Mean weight: {comparison['weight_center'].mean():.2f}")
    print(f"  Parcels with weight >= 0.8: {(comparison['weight_center'] >= 0.8).sum()} (very small)")
    print(f"  Parcels with weight ~0.5: {((comparison['weight_center'] >= 0.4) & (comparison['weight_center'] <= 0.6)).sum()} (small)")
    print(f"  Parcels with weight <= 0.2: {(comparison['weight_center'] <= 0.2).sum()} (larger)")
    
    print("\n5. EXAMPLE PARCELS (10 smallest)")
    print("-" * 50)
    
    examples = comparison.nsmallest(10, PARCEL_ACREAGE_FIELD)
    
    for idx, row in examples.iterrows():
        print(f"\n{row[PARCEL_ID_FIELD]}:")
        print(f"  Actual: {row[PARCEL_ACREAGE_FIELD]:.4f} acres")
        print(f"  Standard: {row['standard_pixels']:.0f} pixels = {row['standard_acres']:.4f} acres (error: {row['standard_error_pct']:+.1f}%)")
        print(f"  Exact: {row['total_pixels']:.2f} pixels = {row['exact_acres']:.4f} acres (error: {row['exact_error_pct']:+.1f}%)")
        print(f"  Details: whole={row['whole_pixels']:.0f}, center={row['center_pixels']:.0f}, weight={row['weight_center']:.2f}")
    
    # Save results
    output_file = Path("outputs") / "exact_method_comparison.csv"
    output_file.parent.mkdir(exist_ok=True)
    comparison.to_csv(output_file, index=False)
    
    print("\n6. OUTPUT")
    print("-" * 50)
    print(f"Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if comparison['exact_error'].abs().mean() < comparison['standard_error'].abs().mean():
        pct_improvement = ((comparison['standard_error'].abs().mean() - 
                           comparison['exact_error'].abs().mean()) / 
                          comparison['standard_error'].abs().mean() * 100)
        print(f"✓ Exact/weighted method reduces mean error by {pct_improvement:.1f}%")
        print(f"✓ Most effective for parcels < {pixel_area_acres:.3f} acres (1 pixel)")
    else:
        print("✗ Exact/weighted method did not improve overall accuracy")
    
    print("\nRecommendation:")
    print("- Use weighted method for parcels < 1 acre")
    print("- Use standard method for larger parcels")
    print("- Consider pixel size (30m) when interpreting results for small parcels")
    
    print("\n" + "=" * 80)
    
    return comparison

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test exact/weighted pixel method")
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of parcels to test (default: 50)"
    )
    
    args = parser.parse_args()
    
    try:
        results = test_exact_method(sample_size=args.sample)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)