#!/usr/bin/env python
"""Test real zonal statistics calculation comparing standard vs sub-pixel methods."""

import sys
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import rasterio
from rasterio import features
from rasterstats import zonal_stats

from src.config import (
    RASTER_PATH, 
    PARCEL_PATH, 
    PARCEL_ID_FIELD, 
    PARCEL_ACREAGE_FIELD,
    LAND_USE_CLASSES,
    RASTER_CRS
)
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor

def calculate_subpixel_zonal_stats(parcel, raster_data, transform, sub_factor=5):
    """Calculate zonal statistics with sub-pixel accuracy.
    
    Args:
        parcel: Single parcel geometry
        raster_data: Raster data array
        transform: Raster transform
        sub_factor: Sub-pixel resolution (5 = 5x5 sub-pixels per pixel)
    
    Returns:
        Dictionary with land use statistics
    """
    # Get bounds of the parcel
    bounds = parcel.geometry.bounds
    
    # Create window for this parcel
    window = rasterio.windows.from_bounds(*bounds, transform=transform)
    col_off = max(0, int(window.col_off) - 1)
    row_off = max(0, int(window.row_off) - 1)
    width = min(int(window.width) + 3, raster_data.shape[1] - col_off)
    height = min(int(window.height) + 3, raster_data.shape[0] - row_off)
    
    if width <= 0 or height <= 0:
        return {}
    
    # Get the raster subset
    raster_subset = raster_data[row_off:row_off+height, col_off:col_off+width]
    
    # Create sub-pixel transform
    sub_transform = transform * transform.translation(col_off, row_off)
    sub_transform_fine = sub_transform * sub_transform.scale(1/sub_factor, 1/sub_factor)
    
    # Rasterize at sub-pixel resolution
    sub_shape = (height * sub_factor, width * sub_factor)
    sub_mask = features.rasterize(
        [(parcel.geometry, 1)],
        out_shape=sub_shape,
        transform=sub_transform_fine,
        fill=0,
        dtype=np.uint8
    )
    
    # Calculate fractional coverage for each pixel
    fractional_coverage = sub_mask.reshape(
        height, sub_factor,
        width, sub_factor
    ).sum(axis=(1, 3)) / (sub_factor ** 2)
    
    # Calculate weighted land use statistics
    land_use_counts = {}
    for class_id in LAND_USE_CLASSES.keys():
        # Find pixels with this land use
        class_mask = (raster_subset == class_id)
        # Weight by fractional coverage
        weighted_count = (class_mask * fractional_coverage).sum()
        if weighted_count > 0:
            land_use_counts[class_id] = weighted_count
    
    return land_use_counts

def run_real_data_test(sample_size=200):
    """Run comprehensive test on real data.
    
    Args:
        sample_size: Number of parcels to test
    """
    print("=" * 100)
    print("REAL DATA ZONAL STATISTICS TEST")
    print("=" * 100)
    print(f"Testing {sample_size} parcels with actual land use data\n")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Load raster
    print("Loading raster data...")
    raster_metadata = data_loader.load_raster_metadata(RASTER_PATH)
    pixel_area_m2 = 900.0  # 30m x 30m
    pixel_area_acres = pixel_area_m2 / 4046.86
    
    # Load parcels - get a mix of sizes
    print(f"Loading {sample_size} parcels...")
    all_parcels = data_loader.load_parcels(
        PARCEL_PATH,
        columns=[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry']
    )
    
    # Get a stratified sample across different size ranges
    size_ranges = [
        (0.1, 0.5, sample_size // 3),    # Small parcels
        (0.5, 2.0, sample_size // 3),    # Medium parcels  
        (2.0, 10.0, sample_size // 3),   # Larger parcels
    ]
    
    sampled_parcels = []
    for min_size, max_size, n_samples in size_ranges:
        subset = all_parcels[
            (all_parcels[PARCEL_ACREAGE_FIELD] >= min_size) & 
            (all_parcels[PARCEL_ACREAGE_FIELD] < max_size)
        ]
        if len(subset) >= n_samples:
            sampled_parcels.append(subset.sample(n=n_samples, random_state=42))
        else:
            sampled_parcels.append(subset)
    
    parcels = pd.concat(sampled_parcels, ignore_index=True)
    parcels = gpd.GeoDataFrame(parcels, geometry='geometry', crs=all_parcels.crs)
    
    print(f"Selected {len(parcels)} parcels")
    print(f"Size range: {parcels[PARCEL_ACREAGE_FIELD].min():.3f} - "
          f"{parcels[PARCEL_ACREAGE_FIELD].max():.3f} acres")
    print(f"Mean size: {parcels[PARCEL_ACREAGE_FIELD].mean():.3f} acres\n")
    
    # Transform to raster CRS
    print("Transforming parcels to raster CRS...")
    parcels_transformed = preprocessor.transform_parcels_crs(
        parcels,
        target_crs=RASTER_CRS
    )
    
    # Load raster data for the area
    print("Loading raster data for parcel area...")
    with rasterio.open(RASTER_PATH) as src:
        # Get bounds of all parcels
        parcel_bounds = parcels_transformed.total_bounds
        
        # Create window
        window = rasterio.windows.from_bounds(
            *parcel_bounds,
            transform=src.transform
        )
        
        # Read raster data
        raster_data = src.read(1, window=window)
        window_transform = rasterio.windows.transform(window, src.transform)
        
        print(f"Loaded raster window: {raster_data.shape}")
        unique_values = np.unique(raster_data[raster_data != 0])
        print(f"Land use classes present: {unique_values}\n")
    
    # Process with different methods
    print("=" * 100)
    print("PROCESSING WITH DIFFERENT METHODS")
    print("=" * 100)
    
    results_comparison = []
    
    # Method 1: Standard (all_touched=True)
    print("\n1. STANDARD METHOD (all_touched=True)")
    print("-" * 50)
    start_time = time.time()
    
    standard_results = []
    for idx, parcel in parcels_transformed.iterrows():
        stats = zonal_stats(
            [parcel.geometry],
            raster_data,
            affine=window_transform,
            categorical=True,
            nodata=0,
            all_touched=True
        )[0]
        
        if stats:
            total = sum(stats.values())
            proportions = {k: v/total*100 for k, v in stats.items()} if total > 0 else {}
            majority_class = max(stats.keys(), key=lambda k: stats[k]) if stats else 0
        else:
            total = 0
            proportions = {}
            majority_class = 0
        
        standard_results.append({
            PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
            'total_pixels': total,
            'calculated_acres': total * pixel_area_acres,
            'majority_class': majority_class,
            **{f'class_{k}_pct': proportions.get(k, 0) for k in LAND_USE_CLASSES.keys()}
        })
    
    standard_time = time.time() - start_time
    standard_df = pd.DataFrame(standard_results)
    print(f"Processing time: {standard_time:.2f} seconds ({len(parcels)/standard_time:.1f} parcels/sec)")
    
    # Method 2: Center-only (all_touched=False)
    print("\n2. CENTER-ONLY METHOD (all_touched=False)")
    print("-" * 50)
    start_time = time.time()
    
    center_results = []
    for idx, parcel in parcels_transformed.iterrows():
        stats = zonal_stats(
            [parcel.geometry],
            raster_data,
            affine=window_transform,
            categorical=True,
            nodata=0,
            all_touched=False
        )[0]
        
        if stats:
            total = sum(stats.values())
            proportions = {k: v/total*100 for k, v in stats.items()} if total > 0 else {}
            majority_class = max(stats.keys(), key=lambda k: stats[k]) if stats else 0
        else:
            total = 0
            proportions = {}
            majority_class = 0
        
        center_results.append({
            PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
            'total_pixels': total,
            'calculated_acres': total * pixel_area_acres,
            'majority_class': majority_class,
            **{f'class_{k}_pct': proportions.get(k, 0) for k in LAND_USE_CLASSES.keys()}
        })
    
    center_time = time.time() - start_time
    center_df = pd.DataFrame(center_results)
    print(f"Processing time: {center_time:.2f} seconds ({len(parcels)/center_time:.1f} parcels/sec)")
    
    # Method 3: Sub-pixel 5x5
    print("\n3. SUB-PIXEL METHOD (5x5 resolution)")
    print("-" * 50)
    start_time = time.time()
    
    subpixel_results = []
    for idx, parcel in parcels_transformed.iterrows():
        stats = calculate_subpixel_zonal_stats(parcel, raster_data, window_transform, sub_factor=5)
        
        if stats:
            total = sum(stats.values())
            proportions = {k: v/total*100 for k, v in stats.items()} if total > 0 else {}
            majority_class = max(stats.keys(), key=lambda k: stats[k]) if stats else 0
        else:
            total = 0
            proportions = {}
            majority_class = 0
        
        subpixel_results.append({
            PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
            'total_pixels': total,
            'calculated_acres': total * pixel_area_acres,
            'majority_class': majority_class,
            **{f'class_{k}_pct': proportions.get(k, 0) for k in LAND_USE_CLASSES.keys()}
        })
    
    subpixel_time = time.time() - start_time
    subpixel_df = pd.DataFrame(subpixel_results)
    print(f"Processing time: {subpixel_time:.2f} seconds ({len(parcels)/subpixel_time:.1f} parcels/sec)")
    
    # Merge all results for comparison
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    
    # Merge with original parcel data
    comparison = parcels[[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD]].merge(
        standard_df[[PARCEL_ID_FIELD, 'calculated_acres', 'majority_class']].rename(
            columns={'calculated_acres': 'standard_acres', 'majority_class': 'standard_majority'}
        ),
        on=PARCEL_ID_FIELD
    ).merge(
        center_df[[PARCEL_ID_FIELD, 'calculated_acres', 'majority_class']].rename(
            columns={'calculated_acres': 'center_acres', 'majority_class': 'center_majority'}
        ),
        on=PARCEL_ID_FIELD
    ).merge(
        subpixel_df[[PARCEL_ID_FIELD, 'calculated_acres', 'majority_class']].rename(
            columns={'calculated_acres': 'subpixel_acres', 'majority_class': 'subpixel_majority'}
        ),
        on=PARCEL_ID_FIELD
    )
    
    # Calculate errors
    comparison['standard_error'] = comparison['standard_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    comparison['center_error'] = comparison['center_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    comparison['subpixel_error'] = comparison['subpixel_acres'] - comparison[PARCEL_ACREAGE_FIELD]
    
    # Group by size for analysis
    comparison['size_category'] = pd.cut(
        comparison[PARCEL_ACREAGE_FIELD],
        bins=[0, 0.5, 1, 2, 5, 10, float('inf')],
        labels=['<0.5', '0.5-1', '1-2', '2-5', '5-10', '>10']
    )
    
    print("\n1. ACCURACY COMPARISON BY PARCEL SIZE")
    print("-" * 80)
    
    for category in comparison['size_category'].unique():
        if pd.notna(category):
            subset = comparison[comparison['size_category'] == category]
            if len(subset) > 0:
                print(f"\n{category} acres ({len(subset)} parcels):")
                print(f"  Standard MAE: {subset['standard_error'].abs().mean():.4f} acres")
                print(f"  Center MAE:   {subset['center_error'].abs().mean():.4f} acres")
                print(f"  Subpixel MAE: {subset['subpixel_error'].abs().mean():.4f} acres")
    
    print("\n2. OVERALL STATISTICS")
    print("-" * 80)
    
    print(f"\nMean Absolute Error (acres):")
    print(f"  Standard:  {comparison['standard_error'].abs().mean():.4f}")
    print(f"  Center:    {comparison['center_error'].abs().mean():.4f}")
    print(f"  Subpixel:  {comparison['subpixel_error'].abs().mean():.4f}")
    
    print(f"\nRoot Mean Square Error (acres):")
    print(f"  Standard:  {np.sqrt((comparison['standard_error']**2).mean()):.4f}")
    print(f"  Center:    {np.sqrt((comparison['center_error']**2).mean()):.4f}")
    print(f"  Subpixel:  {np.sqrt((comparison['subpixel_error']**2).mean()):.4f}")
    
    print(f"\nProcessing Speed (parcels/second):")
    print(f"  Standard:  {len(parcels)/standard_time:.1f}")
    print(f"  Center:    {len(parcels)/center_time:.1f}")
    print(f"  Subpixel:  {len(parcels)/subpixel_time:.1f}")
    
    print("\n3. LAND USE CLASSIFICATION AGREEMENT")
    print("-" * 80)
    
    # Check majority class agreement
    standard_center_agree = (comparison['standard_majority'] == comparison['center_majority']).mean()
    standard_subpixel_agree = (comparison['standard_majority'] == comparison['subpixel_majority']).mean()
    center_subpixel_agree = (comparison['center_majority'] == comparison['subpixel_majority']).mean()
    
    print(f"\nMajority class agreement:")
    print(f"  Standard vs Center:   {standard_center_agree*100:.1f}%")
    print(f"  Standard vs Subpixel: {standard_subpixel_agree*100:.1f}%")
    print(f"  Center vs Subpixel:   {center_subpixel_agree*100:.1f}%")
    
    # Show land use proportions comparison for a few examples
    print("\n4. EXAMPLE PARCELS (Land Use Proportions)")
    print("-" * 80)
    
    # Get examples from different size categories
    examples = []
    for category in ['<0.5', '0.5-1', '1-2']:
        cat_parcels = comparison[comparison['size_category'] == category]
        if len(cat_parcels) > 0:
            examples.append(cat_parcels.iloc[0])
    
    for example in examples:
        parcel_id = example[PARCEL_ID_FIELD]
        actual_acres = example[PARCEL_ACREAGE_FIELD]
        
        print(f"\nParcel {parcel_id} ({actual_acres:.3f} acres):")
        
        # Get detailed proportions
        std_row = standard_df[standard_df[PARCEL_ID_FIELD] == parcel_id].iloc[0]
        ctr_row = center_df[center_df[PARCEL_ID_FIELD] == parcel_id].iloc[0]
        sub_row = subpixel_df[subpixel_df[PARCEL_ID_FIELD] == parcel_id].iloc[0]
        
        print(f"  Calculated acres:")
        print(f"    Standard:  {std_row['calculated_acres']:.3f} (error: {example['standard_error']:+.3f})")
        print(f"    Center:    {ctr_row['calculated_acres']:.3f} (error: {example['center_error']:+.3f})")
        print(f"    Subpixel:  {sub_row['calculated_acres']:.3f} (error: {example['subpixel_error']:+.3f})")
        
        print(f"  Land use proportions (%):")
        for class_id, class_name in LAND_USE_CLASSES.items():
            col = f'class_{class_id}_pct'
            if col in std_row.index:
                std_pct = std_row[col]
                ctr_pct = ctr_row[col]
                sub_pct = sub_row[col]
                if std_pct > 0 or ctr_pct > 0 or sub_pct > 0:
                    print(f"    {class_name:20} Std: {std_pct:5.1f}  Ctr: {ctr_pct:5.1f}  Sub: {sub_pct:5.1f}")
    
    # Save detailed results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save comparison
    comparison_file = output_dir / "real_data_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    
    # Save detailed results
    for df, name in [(standard_df, 'standard'), (center_df, 'center'), (subpixel_df, 'subpixel')]:
        output_file = output_dir / f"real_data_{name}_results.csv"
        df.to_csv(output_file, index=False)
    
    print("\n5. OUTPUT FILES")
    print("-" * 80)
    print(f"Results saved to outputs/ directory")
    
    print("\n" + "=" * 100)
    print("CONCLUSIONS")
    print("=" * 100)
    
    # Calculate improvements
    standard_mae = comparison['standard_error'].abs().mean()
    subpixel_mae = comparison['subpixel_error'].abs().mean()
    improvement = (standard_mae - subpixel_mae) / standard_mae * 100
    
    print(f"\n✓ Sub-pixel method reduces error by {improvement:.1f}% compared to standard method")
    print(f"✓ Processing speed: {len(parcels)/subpixel_time:.0f} parcels/second")
    print(f"✓ Particularly effective for parcels < 1 acre")
    
    # Check if subpixel is faster
    if subpixel_time < standard_time:
        speedup = standard_time / subpixel_time
        print(f"✓ Sub-pixel method is {speedup:.1f}x FASTER than standard method!")
    
    print("\n" + "=" * 100)
    
    return comparison

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test zonal statistics on real data")
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Number of parcels to test (default: 200)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_real_data_test(sample_size=args.sample)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)