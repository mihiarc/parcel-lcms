#!/usr/bin/env python
"""Benchmark different fractional pixel calculation methods for accuracy vs compute time tradeoffs."""

import sys
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple
import psutil

import rasterio
from rasterio import features
from rasterstats import zonal_stats
from shapely.geometry import box

# Import pipeline components
from src.config import RASTER_PATH, PARCEL_PATH, PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor

class FractionalMethodsBenchmark:
    """Benchmark different methods for calculating fractional pixel coverage."""
    
    def __init__(self, raster_path: str, pixel_area_m2: float = 900.0):
        """Initialize benchmark."""
        self.raster_path = raster_path
        self.pixel_area_m2 = pixel_area_m2
        self.pixel_area_acres = pixel_area_m2 / 4046.86
        self.results = []
        
    def method_1_standard(self, parcels: gpd.GeoDataFrame, raster_data, transform) -> Tuple[pd.DataFrame, float]:
        """Method 1: Standard all_touched=True (baseline)."""
        start_time = time.time()
        results = []
        
        for idx, parcel in parcels.iterrows():
            stats = zonal_stats(
                [parcel.geometry],
                raster_data,
                affine=transform,
                stats=['count'],
                nodata=0,
                all_touched=True
            )[0]
            
            pixel_count = stats.get('count', 0) if stats else 0
            acres = pixel_count * self.pixel_area_acres
            
            results.append({
                PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
                'pixel_count': pixel_count,
                'calculated_acres': acres
            })
        
        elapsed = time.time() - start_time
        return pd.DataFrame(results), elapsed
    
    def method_2_center_only(self, parcels: gpd.GeoDataFrame, raster_data, transform) -> Tuple[pd.DataFrame, float]:
        """Method 2: Center-only (all_touched=False)."""
        start_time = time.time()
        results = []
        
        for idx, parcel in parcels.iterrows():
            stats = zonal_stats(
                [parcel.geometry],
                raster_data,
                affine=transform,
                stats=['count'],
                nodata=0,
                all_touched=False
            )[0]
            
            pixel_count = stats.get('count', 0) if stats else 0
            acres = pixel_count * self.pixel_area_acres
            
            results.append({
                PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
                'pixel_count': pixel_count,
                'calculated_acres': acres
            })
        
        elapsed = time.time() - start_time
        return pd.DataFrame(results), elapsed
    
    def method_3_weighted(self, parcels: gpd.GeoDataFrame, raster_data, transform) -> Tuple[pd.DataFrame, float]:
        """Method 3: Weighted average of all_touched and center."""
        start_time = time.time()
        results = []
        
        for idx, parcel in parcels.iterrows():
            # Get both statistics
            stats_all = zonal_stats(
                [parcel.geometry],
                raster_data,
                affine=transform,
                stats=['count'],
                nodata=0,
                all_touched=True
            )[0]
            
            stats_center = zonal_stats(
                [parcel.geometry],
                raster_data,
                affine=transform,
                stats=['count'],
                nodata=0,
                all_touched=False
            )[0]
            
            all_count = stats_all.get('count', 0) if stats_all else 0
            center_count = stats_center.get('count', 0) if stats_center else 0
            
            # Calculate weight based on parcel size
            parcel_acres = parcel.get(PARCEL_ACREAGE_FIELD, 1.0)
            if parcel_acres < self.pixel_area_acres:
                weight_center = 0.8
            elif parcel_acres < 2 * self.pixel_area_acres:
                weight_center = 0.5
            else:
                weight_center = 0.2
            
            weighted_count = all_count * (1 - weight_center) + center_count * weight_center
            acres = weighted_count * self.pixel_area_acres
            
            results.append({
                PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
                'pixel_count': weighted_count,
                'calculated_acres': acres
            })
        
        elapsed = time.time() - start_time
        return pd.DataFrame(results), elapsed
    
    def method_4_subpixel_2x2(self, parcels: gpd.GeoDataFrame, raster_data, transform) -> Tuple[pd.DataFrame, float]:
        """Method 4: 2x2 sub-pixel rasterization (4 sub-pixels per pixel)."""
        start_time = time.time()
        results = []
        sub_factor = 2
        
        for idx, parcel in parcels.iterrows():
            # Get bounds of the parcel
            bounds = parcel.geometry.bounds
            
            # Create window for this parcel
            window = rasterio.windows.from_bounds(*bounds, transform=transform)
            col_off = int(window.col_off)
            row_off = int(window.row_off)
            width = int(window.width) + 2
            height = int(window.height) + 2
            
            # Ensure within raster bounds
            col_off = max(0, col_off)
            row_off = max(0, row_off)
            width = min(width, raster_data.shape[1] - col_off)
            height = min(height, raster_data.shape[0] - row_off)
            
            if width <= 0 or height <= 0:
                pixel_count = 0
            else:
                # Create sub-pixel transform
                sub_transform = transform * transform.translation(col_off, row_off)
                sub_transform = sub_transform * sub_transform.scale(1/sub_factor, 1/sub_factor)
                
                # Rasterize at sub-pixel resolution
                sub_shape = (height * sub_factor, width * sub_factor)
                sub_mask = features.rasterize(
                    [(parcel.geometry, 1)],
                    out_shape=sub_shape,
                    transform=sub_transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                # Calculate fractional coverage
                pixel_count = sub_mask.sum() / (sub_factor ** 2)
            
            acres = pixel_count * self.pixel_area_acres
            
            results.append({
                PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
                'pixel_count': pixel_count,
                'calculated_acres': acres
            })
        
        elapsed = time.time() - start_time
        return pd.DataFrame(results), elapsed
    
    def method_5_subpixel_5x5(self, parcels: gpd.GeoDataFrame, raster_data, transform) -> Tuple[pd.DataFrame, float]:
        """Method 5: 5x5 sub-pixel rasterization (25 sub-pixels per pixel)."""
        start_time = time.time()
        results = []
        sub_factor = 5
        
        for idx, parcel in parcels.iterrows():
            bounds = parcel.geometry.bounds
            window = rasterio.windows.from_bounds(*bounds, transform=transform)
            col_off = int(window.col_off)
            row_off = int(window.row_off)
            width = int(window.width) + 2
            height = int(window.height) + 2
            
            col_off = max(0, col_off)
            row_off = max(0, row_off)
            width = min(width, raster_data.shape[1] - col_off)
            height = min(height, raster_data.shape[0] - row_off)
            
            if width <= 0 or height <= 0:
                pixel_count = 0
            else:
                sub_transform = transform * transform.translation(col_off, row_off)
                sub_transform = sub_transform * sub_transform.scale(1/sub_factor, 1/sub_factor)
                
                sub_shape = (height * sub_factor, width * sub_factor)
                sub_mask = features.rasterize(
                    [(parcel.geometry, 1)],
                    out_shape=sub_shape,
                    transform=sub_transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                pixel_count = sub_mask.sum() / (sub_factor ** 2)
            
            acres = pixel_count * self.pixel_area_acres
            
            results.append({
                PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
                'pixel_count': pixel_count,
                'calculated_acres': acres
            })
        
        elapsed = time.time() - start_time
        return pd.DataFrame(results), elapsed
    
    def method_6_subpixel_10x10(self, parcels: gpd.GeoDataFrame, raster_data, transform) -> Tuple[pd.DataFrame, float]:
        """Method 6: 10x10 sub-pixel rasterization (100 sub-pixels per pixel)."""
        start_time = time.time()
        results = []
        sub_factor = 10
        
        for idx, parcel in parcels.iterrows():
            bounds = parcel.geometry.bounds
            window = rasterio.windows.from_bounds(*bounds, transform=transform)
            col_off = int(window.col_off)
            row_off = int(window.row_off)
            width = int(window.width) + 2
            height = int(window.height) + 2
            
            col_off = max(0, col_off)
            row_off = max(0, row_off)
            width = min(width, raster_data.shape[1] - col_off)
            height = min(height, raster_data.shape[0] - row_off)
            
            if width <= 0 or height <= 0:
                pixel_count = 0
            else:
                sub_transform = transform * transform.translation(col_off, row_off)
                sub_transform = sub_transform * sub_transform.scale(1/sub_factor, 1/sub_factor)
                
                sub_shape = (height * sub_factor, width * sub_factor)
                sub_mask = features.rasterize(
                    [(parcel.geometry, 1)],
                    out_shape=sub_shape,
                    transform=sub_transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                pixel_count = sub_mask.sum() / (sub_factor ** 2)
            
            acres = pixel_count * self.pixel_area_acres
            
            results.append({
                PARCEL_ID_FIELD: parcel[PARCEL_ID_FIELD],
                'pixel_count': pixel_count,
                'calculated_acres': acres
            })
        
        elapsed = time.time() - start_time
        return pd.DataFrame(results), elapsed

def run_benchmark(sample_sizes=[10, 50, 100]):
    """Run comprehensive benchmark of all methods."""
    print("=" * 100)
    print("FRACTIONAL PIXEL METHODS BENCHMARK")
    print("=" * 100)
    print()
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Load raster metadata
    print("Loading raster metadata...")
    raster_metadata = data_loader.load_raster_metadata(RASTER_PATH)
    
    # Initialize benchmark
    benchmark = FractionalMethodsBenchmark(RASTER_PATH)
    
    # Define methods to test
    methods = [
        ("Standard (all_touched)", benchmark.method_1_standard),
        ("Center-only", benchmark.method_2_center_only),
        ("Weighted", benchmark.method_3_weighted),
        ("Sub-pixel 2x2", benchmark.method_4_subpixel_2x2),
        ("Sub-pixel 5x5", benchmark.method_5_subpixel_5x5),
        ("Sub-pixel 10x10", benchmark.method_6_subpixel_10x10),
    ]
    
    all_results = []
    
    for sample_size in sample_sizes:
        print(f"\n{'=' * 100}")
        print(f"TESTING WITH {sample_size} PARCELS")
        print("=" * 100)
        
        # Load parcels
        print(f"\nLoading {sample_size} small parcels...")
        all_parcels = data_loader.load_parcels(
            PARCEL_PATH,
            columns=[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'geometry']
        )
        
        # Get parcels in the 0.1-1.0 acre range for meaningful testing
        # Filter for parcels between 0.1 and 1.0 acres
        small_parcels = all_parcels[
            (all_parcels[PARCEL_ACREAGE_FIELD] > 0.1) & 
            (all_parcels[PARCEL_ACREAGE_FIELD] < 1.0)
        ]
        
        if len(small_parcels) < sample_size:
            print(f"Warning: Only {len(small_parcels)} parcels in 0.1-1.0 acre range")
            parcels = small_parcels
        else:
            # Sample from the small parcels
            parcels = small_parcels.sample(n=sample_size, random_state=42)
        
        print(f"Parcel size range: {parcels[PARCEL_ACREAGE_FIELD].min():.6f} - "
              f"{parcels[PARCEL_ACREAGE_FIELD].max():.6f} acres")
        
        # Transform to raster CRS
        print("Transforming to raster CRS...")
        parcels_transformed = preprocessor.transform_parcels_crs(
            parcels,
            target_crs=raster_metadata['crs']
        )
        
        # Calculate true area in transformed CRS
        parcels_transformed['true_acres'] = parcels_transformed.geometry.area / 4046.86
        
        # Load raster window
        with rasterio.open(RASTER_PATH) as src:
            parcel_bounds = parcels_transformed.total_bounds
            window = rasterio.windows.from_bounds(*parcel_bounds, transform=src.transform)
            raster_data = src.read(1, window=window)
            window_transform = rasterio.windows.transform(window, src.transform)
        
        # Test each method
        method_results = []
        
        for method_name, method_func in methods:
            print(f"\nTesting {method_name}...")
            
            # Get memory before
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run method
            results_df, elapsed_time = method_func(parcels_transformed, raster_data, window_transform)
            
            # Get memory after
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            # Calculate accuracy metrics
            results_df = results_df.merge(
                parcels_transformed[[PARCEL_ID_FIELD, PARCEL_ACREAGE_FIELD, 'true_acres']],
                on=PARCEL_ID_FIELD
            )
            
            results_df['error_acres'] = results_df['calculated_acres'] - results_df[PARCEL_ACREAGE_FIELD]
            results_df['abs_error'] = results_df['error_acres'].abs()
            
            # Handle division by zero for percentage error
            results_df['error_pct'] = np.where(
                results_df[PARCEL_ACREAGE_FIELD] > 0,
                (results_df['error_acres'] / results_df[PARCEL_ACREAGE_FIELD]) * 100,
                np.where(results_df['calculated_acres'] > 0, np.inf, 0)
            )
            
            # Calculate metrics
            mae = results_df['abs_error'].mean()
            rmse = np.sqrt((results_df['error_acres'] ** 2).mean())
            max_error = results_df['abs_error'].max()
            
            # Calculate correlation with true acres
            valid_mask = (results_df[PARCEL_ACREAGE_FIELD] > 0) & (results_df['calculated_acres'] > 0)
            if valid_mask.any():
                correlation = results_df[valid_mask][PARCEL_ACREAGE_FIELD].corr(
                    results_df[valid_mask]['calculated_acres']
                )
            else:
                correlation = 0
            
            # Store results
            method_result = {
                'method': method_name,
                'sample_size': sample_size,
                'elapsed_time': elapsed_time,
                'parcels_per_second': sample_size / elapsed_time,
                'memory_mb': mem_used,
                'mae_acres': mae,
                'rmse_acres': rmse,
                'max_error_acres': max_error,
                'correlation': correlation,
            }
            
            method_results.append(method_result)
            
            # Print immediate results
            print(f"  Time: {elapsed_time:.3f} seconds ({sample_size/elapsed_time:.1f} parcels/sec)")
            print(f"  Memory: {mem_used:.1f} MB")
            print(f"  MAE: {mae:.6f} acres")
            print(f"  RMSE: {rmse:.6f} acres")
            print(f"  Correlation: {correlation:.4f}")
        
        all_results.extend(method_results)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # Print comprehensive summary
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Performance comparison
    print("\n1. PERFORMANCE METRICS")
    print("-" * 80)
    
    for sample_size in sample_sizes:
        print(f"\nSample size: {sample_size} parcels")
        subset = summary_df[summary_df['sample_size'] == sample_size]
        
        print(f"{'Method':<20} {'Time (s)':<12} {'Parcels/s':<12} {'Memory (MB)':<12}")
        print("-" * 56)
        
        for _, row in subset.iterrows():
            print(f"{row['method']:<20} {row['elapsed_time']:<12.3f} "
                  f"{row['parcels_per_second']:<12.1f} {row['memory_mb']:<12.1f}")
    
    # Accuracy comparison
    print("\n2. ACCURACY METRICS")
    print("-" * 80)
    
    for sample_size in sample_sizes:
        print(f"\nSample size: {sample_size} parcels")
        subset = summary_df[summary_df['sample_size'] == sample_size]
        
        print(f"{'Method':<20} {'MAE (acres)':<15} {'RMSE (acres)':<15} {'Correlation':<12}")
        print("-" * 62)
        
        for _, row in subset.iterrows():
            print(f"{row['method']:<20} {row['mae_acres']:<15.6f} "
                  f"{row['rmse_acres']:<15.6f} {row['correlation']:<12.4f}")
    
    # Calculate speedup and accuracy improvement
    print("\n3. TRADEOFF ANALYSIS")
    print("-" * 80)
    
    # Use largest sample size for final analysis
    final_sample = summary_df[summary_df['sample_size'] == sample_sizes[-1]]
    baseline = final_sample[final_sample['method'] == 'Standard (all_touched)'].iloc[0]
    
    print(f"\nCompared to baseline (Standard all_touched):")
    print(f"{'Method':<20} {'Speedup':<12} {'Accuracy Gain':<15} {'Efficiency':<15}")
    print(f"{'':20} {'(x faster)':<12} {'(% less error)':<15} {'(gain/slowdown)':<15}")
    print("-" * 62)
    
    for _, row in final_sample.iterrows():
        if row['method'] != 'Standard (all_touched)':
            speedup = baseline['parcels_per_second'] / row['parcels_per_second']
            accuracy_gain = ((baseline['mae_acres'] - row['mae_acres']) / baseline['mae_acres']) * 100
            efficiency = accuracy_gain / speedup if speedup > 0 else 0
            
            print(f"{row['method']:<20} {1/speedup:<12.2f}x "
                  f"{accuracy_gain:<15.1f}% {efficiency:<15.1f}")
    
    # Recommendations
    print("\n4. RECOMMENDATIONS")
    print("-" * 80)
    
    print("\nBased on accuracy vs compute time tradeoffs:")
    
    # Find best methods for different scenarios
    fast_accurate = final_sample[
        (final_sample['parcels_per_second'] > 50) & 
        (final_sample['mae_acres'] < baseline['mae_acres'] * 0.5)
    ]
    
    if not fast_accurate.empty:
        best_balanced = fast_accurate.iloc[0]
        print(f"\n✓ BEST BALANCED: {best_balanced['method']}")
        print(f"  - {best_balanced['parcels_per_second']:.0f} parcels/second")
        print(f"  - {(1 - best_balanced['mae_acres']/baseline['mae_acres'])*100:.0f}% error reduction")
    
    most_accurate = final_sample.nsmallest(1, 'mae_acres').iloc[0]
    print(f"\n✓ MOST ACCURATE: {most_accurate['method']}")
    print(f"  - {most_accurate['parcels_per_second']:.0f} parcels/second")
    print(f"  - {(1 - most_accurate['mae_acres']/baseline['mae_acres'])*100:.0f}% error reduction")
    
    fastest = final_sample.nlargest(1, 'parcels_per_second').iloc[0]
    print(f"\n✓ FASTEST: {fastest['method']}")
    print(f"  - {fastest['parcels_per_second']:.0f} parcels/second")
    
    # Processing time estimates for 2M parcels
    print("\n5. SCALING ESTIMATES (for 2 million parcels)")
    print("-" * 80)
    
    for _, row in final_sample.iterrows():
        hours = 2_000_000 / row['parcels_per_second'] / 3600
        print(f"{row['method']:<20} {hours:>8.1f} hours")
    
    # Save detailed results
    output_file = Path("outputs") / "fractional_methods_benchmark.csv"
    output_file.parent.mkdir(exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n6. OUTPUT")
    print("-" * 80)
    print(f"Detailed results saved to: {output_file}")
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)
    
    return summary_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark fractional pixel methods")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 50, 100],
        help="Sample sizes to test (default: 10 50 100)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark(sample_sizes=args.sizes)
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)