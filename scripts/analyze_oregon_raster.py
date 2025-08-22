#!/usr/bin/env python
"""Analyze raster statistics for Oregon only and compare with Oregon parcels."""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path
import json
from shapely.geometry import mapping

def get_oregon_boundary():
    """Load Oregon state boundary from US states shapefile."""
    
    states_path = Path('data/cb_2018_us_state_20m/cb_2018_us_state_20m.shp')
    if not states_path.exists():
        raise FileNotFoundError(f"States shapefile not found: {states_path}")
    
    print(f"Loading states shapefile: {states_path}")
    states = gpd.read_file(states_path)
    
    # Filter for Oregon (STUSPS = 'OR' or NAME = 'Oregon')
    oregon = states[(states['STUSPS'] == 'OR') | (states['NAME'] == 'Oregon')]
    
    if oregon.empty:
        print("Available states:", states['NAME'].unique())
        raise ValueError("Oregon not found in states shapefile")
    
    print(f"Found Oregon with area: {oregon.geometry.area.values[0]:.2f} sq degrees")
    
    return oregon

def analyze_oregon_raster(raster_path, oregon_gdf):
    """Analyze raster pixels within Oregon boundaries."""
    
    print(f"\nAnalyzing raster within Oregon boundaries...")
    
    with rasterio.open(raster_path) as src:
        # Get raster metadata
        pixel_width = abs(src.transform[0])
        pixel_height = abs(src.transform[4])
        pixel_area_m2 = pixel_width * pixel_height
        pixel_area_acres = pixel_area_m2 / 4046.86
        
        print(f"Raster CRS: {src.crs}")
        print(f"Pixel size: {pixel_width}m x {pixel_height}m")
        print(f"Pixel area: {pixel_area_acres:.4f} acres")
        
        # Reproject Oregon boundary to raster CRS if needed
        if oregon_gdf.crs != src.crs:
            print(f"Reprojecting Oregon boundary from {oregon_gdf.crs} to {src.crs}")
            oregon_reprojected = oregon_gdf.to_crs(src.crs)
        else:
            oregon_reprojected = oregon_gdf
        
        # Get Oregon geometry
        oregon_geom = oregon_reprojected.geometry.values[0]
        
        # Mask raster with Oregon boundary
        print("Masking raster with Oregon boundary...")
        oregon_data, oregon_transform = mask(src, [oregon_geom], crop=True, nodata=0)
        oregon_data = oregon_data[0]  # Get first band
        
        print(f"Oregon raster shape: {oregon_data.shape}")
        print(f"Oregon raster size: {oregon_data.size:,} pixels")
        
        # Count pixels by value
        unique, counts = np.unique(oregon_data, return_counts=True)
        pixel_counts = dict(zip(unique, counts))
        
        # Calculate total and valid pixels
        total_pixels = oregon_data.size
        valid_pixels = total_pixels - pixel_counts.get(0, 0)
        
        print(f"Total pixels in Oregon: {total_pixels:,}")
        print(f"Valid pixels in Oregon: {valid_pixels:,}")
        print(f"NoData pixels: {pixel_counts.get(0, 0):,}")
    
    return pixel_counts, pixel_area_acres, total_pixels, valid_pixels

def map_lcms_to_land_use(pixel_counts):
    """Map LCMS pixel values to our 5 land use classes."""
    
    # LCMS to land use mapping
    lcms_mapping = {
        1: 'DEVELOPED',
        2: 'AGRICULTURE',
        3: 'RANGELAND_PASTURE',
        4: 'FOREST',
        5: 'OTHER',
        6: 'OTHER',
        7: 'OTHER',
        8: 'DEVELOPED',
        9: 'FOREST',
        10: 'RANGELAND_PASTURE',
        11: 'AGRICULTURE',
        12: 'AGRICULTURE',
        13: 'OTHER',
        14: 'OTHER',
        15: 'OTHER'
    }
    
    # Aggregate by land use class
    land_use_pixels = {
        'AGRICULTURE': 0,
        'DEVELOPED': 0,
        'FOREST': 0,
        'OTHER': 0,
        'RANGELAND_PASTURE': 0
    }
    
    for pixel_val, count in pixel_counts.items():
        if pixel_val > 0 and pixel_val in lcms_mapping:
            land_use = lcms_mapping[pixel_val]
            land_use_pixels[land_use] += count
    
    return land_use_pixels

def main():
    # Paths
    raster_path = Path('data/LCMS_CONUS_v2024-10_Land_Use_2024.tif')
    parquet_files = list(Path('outputs').glob('parcel_land_use_results_*.parquet'))
    
    if not raster_path.exists():
        print(f"Raster file not found: {raster_path}")
        return
    
    if not parquet_files:
        print("No parquet files found in outputs/")
        return
    
    latest_parcel_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    
    # Get Oregon boundary
    print("="*80)
    print("OREGON RASTER ANALYSIS")
    print("="*80)
    
    oregon_gdf = get_oregon_boundary()
    
    # Analyze Oregon raster
    pixel_counts, pixel_area_acres, total_pixels, valid_pixels = analyze_oregon_raster(
        raster_path, oregon_gdf
    )
    
    # Map to land use classes
    land_use_pixels = map_lcms_to_land_use(pixel_counts)
    
    # Calculate Oregon raster-based acres
    oregon_raster_acres = {
        land_use: count * pixel_area_acres 
        for land_use, count in land_use_pixels.items()
    }
    total_oregon_raster_acres = sum(oregon_raster_acres.values())
    
    # Calculate Oregon raster percentages
    oregon_raster_percentages = {
        land_use: (acres / total_oregon_raster_acres * 100) if total_oregon_raster_acres > 0 else 0
        for land_use, acres in oregon_raster_acres.items()
    }
    
    print(f"\nOregon Raster Statistics:")
    print(f"  Total valid acres: {total_oregon_raster_acres:,.1f}")
    
    print(f"\nOregon Raster Land Use Distribution:")
    for land_use in sorted(land_use_pixels.keys()):
        pixels = land_use_pixels[land_use]
        acres = oregon_raster_acres[land_use]
        pct = oregon_raster_percentages[land_use]
        print(f"  {land_use:20s}: {pixels:12,} pixels | {acres:15,.1f} acres | {pct:6.2f}%")
    
    # Load parcel data
    print(f"\nLoading Oregon parcel data: {latest_parcel_file}")
    df = pd.read_parquet(latest_parcel_file)
    
    # Calculate parcel-based acres
    df['agriculture_acres'] = df['CAL_ACREAGE'] * (df['agriculture_pct'] / 100)
    df['developed_acres'] = df['CAL_ACREAGE'] * (df['developed_pct'] / 100)
    df['forest_acres'] = df['CAL_ACREAGE'] * (df['forest_pct'] / 100)
    df['other_acres'] = df['CAL_ACREAGE'] * (df['other_pct'] / 100)
    df['rangeland_pasture_acres'] = df['CAL_ACREAGE'] * (df['rangeland_pasture_pct'] / 100)
    
    parcel_acres = {
        'AGRICULTURE': df['agriculture_acres'].sum(),
        'DEVELOPED': df['developed_acres'].sum(),
        'FOREST': df['forest_acres'].sum(),
        'OTHER': df['other_acres'].sum(),
        'RANGELAND_PASTURE': df['rangeland_pasture_acres'].sum()
    }
    
    total_parcel_acres = df['CAL_ACREAGE'].sum()
    
    # Calculate parcel percentages
    parcel_percentages = {
        land_use: (acres / total_parcel_acres * 100) 
        for land_use, acres in parcel_acres.items()
    }
    
    print("\n" + "="*80)
    print("COMPARISON: OREGON RASTER vs OREGON PARCELS")
    print("="*80)
    
    print(f"\nCoverage Comparison:")
    print(f"  Oregon raster acres: {total_oregon_raster_acres:,.1f}")
    print(f"  Oregon parcel acres: {total_parcel_acres:,.1f}")
    print(f"  Parcel coverage: {(total_parcel_acres / total_oregon_raster_acres * 100):.2f}% of Oregon")
    print(f"  Number of parcels: {len(df):,}")
    
    print(f"\n{'Land Use':20s} | {'Oregon Raster %':>15s} | {'Parcel Calc %':>14s} | {'Difference':>12s}")
    print("-"*75)
    
    total_abs_diff = 0
    for land_use in sorted(oregon_raster_acres.keys()):
        raster_pct = oregon_raster_percentages[land_use]
        parcel_pct = parcel_percentages[land_use]
        diff = parcel_pct - raster_pct
        total_abs_diff += abs(diff)
        
        print(f"{land_use:20s} | {raster_pct:15.2f} | {parcel_pct:14.2f} | {diff:+12.2f}")
    
    # Calculate accuracy metrics
    mean_abs_error = total_abs_diff / len(oregon_raster_acres)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean([
        (parcel_percentages[lu] - oregon_raster_percentages[lu])**2 
        for lu in oregon_raster_acres.keys()
    ]))
    
    # Calculate correlation
    raster_vals = [oregon_raster_percentages[lu] for lu in sorted(oregon_raster_acres.keys())]
    parcel_vals = [parcel_percentages[lu] for lu in sorted(oregon_raster_acres.keys())]
    if len(raster_vals) > 1:
        correlation = np.corrcoef(raster_vals, parcel_vals)[0, 1]
    else:
        correlation = 1.0
    
    print(f"\n{'='*80}")
    print("ACCURACY METRICS")
    print("="*80)
    
    print(f"  Mean Absolute Error: {mean_abs_error:.2f} percentage points")
    print(f"  Root Mean Square Error: {rmse:.2f} percentage points")
    print(f"  Correlation: {correlation:.4f}")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print("="*80)
    
    print(f"\n1. PARCEL COVERAGE IN OREGON:")
    print(f"   - Parcels cover {(total_parcel_acres / total_oregon_raster_acres * 100):.1f}% of Oregon")
    print(f"   - Missing {(total_oregon_raster_acres - total_parcel_acres):,.1f} acres")
    print(f"   - Likely public lands: National Forests, BLM, state parks, water bodies")
    
    print(f"\n2. LAND USE ACCURACY:")
    max_diff_lu = max(oregon_raster_acres.keys(), 
                      key=lambda x: abs(parcel_percentages[x] - oregon_raster_percentages[x]))
    max_diff = abs(parcel_percentages[max_diff_lu] - oregon_raster_percentages[max_diff_lu])
    print(f"   - Largest difference: {max_diff_lu} ({max_diff:.2f} percentage points)")
    print(f"   - Mean absolute error: {mean_abs_error:.2f} percentage points")
    
    print(f"\n3. OREGON LANDSCAPE COMPOSITION:")
    dominant = max(oregon_raster_acres.items(), key=lambda x: x[1])
    print(f"   - Dominant land use: {dominant[0]} ({oregon_raster_percentages[dominant[0]]:.1f}%)")
    print(f"   - Forest coverage: {oregon_raster_percentages['FOREST']:.1f}% (Oregon is heavily forested)")
    print(f"   - Developed land: {oregon_raster_percentages['DEVELOPED']:.1f}%")
    
    # Save summary
    summary_file = Path('outputs') / 'oregon_raster_comparison.txt'
    with open(summary_file, 'w') as f:
        f.write("OREGON RASTER vs OREGON PARCELS COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("Oregon Raster (Ground Truth):\n")
        for land_use in sorted(oregon_raster_acres.keys()):
            f.write(f"  {land_use}: {oregon_raster_percentages[land_use]:.2f}% "
                   f"({oregon_raster_acres[land_use]:,.1f} acres)\n")
        
        f.write(f"\nOregon Parcels (Our Processing):\n")
        for land_use in sorted(parcel_acres.keys()):
            f.write(f"  {land_use}: {parcel_percentages[land_use]:.2f}% "
                   f"({parcel_acres[land_use]:,.1f} acres)\n")
        
        f.write(f"\nCoverage: Parcels cover {(total_parcel_acres / total_oregon_raster_acres * 100):.1f}% of Oregon\n")
        f.write(f"Accuracy: {mean_abs_error:.2f} percentage point mean absolute error\n")
        f.write(f"Correlation: {correlation:.4f}\n")
    
    print(f"\nComparison saved to: {summary_file}")

if __name__ == "__main__":
    main()