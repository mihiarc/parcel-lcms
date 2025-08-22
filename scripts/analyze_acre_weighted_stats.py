#!/usr/bin/env python
"""Calculate acre-weighted land use statistics to understand true landscape composition."""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_acre_weighted_stats(df):
    """Calculate acre-weighted land use statistics."""
    
    # Calculate total acres for each land use type
    df['agriculture_acres'] = df['CAL_ACREAGE'] * (df['agriculture_pct'] / 100)
    df['developed_acres'] = df['CAL_ACREAGE'] * (df['developed_pct'] / 100)
    df['forest_acres'] = df['CAL_ACREAGE'] * (df['forest_pct'] / 100)
    df['other_acres'] = df['CAL_ACREAGE'] * (df['other_pct'] / 100)
    df['rangeland_pasture_acres'] = df['CAL_ACREAGE'] * (df['rangeland_pasture_pct'] / 100)
    
    # Calculate totals
    total_acres = df['CAL_ACREAGE'].sum()
    total_parcels = len(df)
    
    # Sum acres by land use
    land_use_acres = {
        'Agriculture': df['agriculture_acres'].sum(),
        'Developed': df['developed_acres'].sum(),
        'Forest': df['forest_acres'].sum(),
        'Other': df['other_acres'].sum(),
        'Rangeland/Pasture': df['rangeland_pasture_acres'].sum()
    }
    
    # Calculate landscape-level percentages (acre-weighted)
    landscape_percentages = {
        land_use: (acres / total_acres * 100) 
        for land_use, acres in land_use_acres.items()
    }
    
    # Calculate parcel-weighted percentages for comparison
    parcel_percentages = {
        'Agriculture': df['agriculture_pct'].mean(),
        'Developed': df['developed_pct'].mean(),
        'Forest': df['forest_pct'].mean(),
        'Other': df['other_pct'].mean(),
        'Rangeland/Pasture': df['rangeland_pasture_pct'].mean()
    }
    
    return {
        'total_acres': total_acres,
        'total_parcels': total_parcels,
        'land_use_acres': land_use_acres,
        'landscape_percentages': landscape_percentages,
        'parcel_percentages': parcel_percentages
    }

def analyze_by_parcel_size(df):
    """Analyze land use by parcel size categories."""
    
    # Define size categories
    size_bins = [0, 0.22, 1, 5, 10, 50, 100, float('inf')]
    size_labels = ['<0.22 acres (sub-pixel)', '0.22-1 acre', '1-5 acres', 
                   '5-10 acres', '10-50 acres', '50-100 acres', '>100 acres']
    
    df['size_category'] = pd.cut(df['CAL_ACREAGE'], bins=size_bins, labels=size_labels)
    
    results = {}
    for category in size_labels:
        subset = df[df['size_category'] == category]
        if len(subset) > 0:
            total_acres = subset['CAL_ACREAGE'].sum()
            results[category] = {
                'parcel_count': len(subset),
                'total_acres': total_acres,
                'avg_parcel_size': subset['CAL_ACREAGE'].mean(),
                'pct_of_total_acres': (total_acres / df['CAL_ACREAGE'].sum()) * 100,
                'developed_pct': (subset['developed_acres'].sum() / total_acres * 100) if total_acres > 0 else 0
            }
    
    return results

def main():
    # Load processed data
    parquet_files = list(Path('outputs').glob('parcel_land_use_results_*.parquet'))
    if not parquet_files:
        print("No parquet files found in outputs/")
        return
    
    # Use most recent file
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df):,} parcels")
    
    # Calculate acre-weighted statistics
    stats = calculate_acre_weighted_stats(df)
    
    # Print detailed summary
    print("\n" + "="*80)
    print("ACRE-WEIGHTED LAND USE ANALYSIS")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total Parcels: {stats['total_parcels']:,}")
    print(f"  Total Acres: {stats['total_acres']:,.1f}")
    print(f"  Average Parcel Size: {stats['total_acres']/stats['total_parcels']:.2f} acres")
    
    print(f"\n{'='*80}")
    print("LANDSCAPE COMPOSITION (Acre-Weighted)")
    print("="*80)
    print("\nTotal Acres by Land Use:")
    for land_use, acres in stats['land_use_acres'].items():
        pct = stats['landscape_percentages'][land_use]
        print(f"  {land_use:20s}: {acres:12,.1f} acres ({pct:5.2f}%)")
    
    print(f"\n{'='*80}")
    print("COMPARISON: Acre-Weighted vs Parcel-Weighted")
    print("="*80)
    print(f"{'Land Use':20s} | {'Acre-Weighted %':>15s} | {'Parcel-Weighted %':>17s} | {'Difference':>10s}")
    print("-"*80)
    
    for land_use in stats['landscape_percentages']:
        acre_pct = stats['landscape_percentages'][land_use]
        parcel_pct = stats['parcel_percentages'][land_use]
        diff = acre_pct - parcel_pct
        print(f"{land_use:20s} | {acre_pct:15.2f} | {parcel_pct:17.2f} | {diff:+10.2f}")
    
    # Analyze by parcel size
    print(f"\n{'='*80}")
    print("LAND USE BY PARCEL SIZE")
    print("="*80)
    
    size_analysis = analyze_by_parcel_size(df)
    
    print(f"\n{'Size Category':25s} | {'Parcels':>10s} | {'Total Acres':>12s} | {'% of Acres':>10s} | {'% Developed':>11s}")
    print("-"*80)
    
    for category, data in size_analysis.items():
        print(f"{category:25s} | {data['parcel_count']:10,} | {data['total_acres']:12,.1f} | "
              f"{data['pct_of_total_acres']:10.2f} | {data['developed_pct']:11.2f}")
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print("="*80)
    
    developed_acres = stats['land_use_acres']['Developed']
    developed_pct = stats['landscape_percentages']['Developed']
    parcel_developed_pct = stats['parcel_percentages']['Developed']
    
    print(f"\n1. DEVELOPMENT ANALYSIS:")
    print(f"   - Total Developed Acres: {developed_acres:,.1f}")
    print(f"   - Landscape Development: {developed_pct:.2f}% (acre-weighted)")
    print(f"   - Parcel-Based Development: {parcel_developed_pct:.2f}% (parcel-weighted)")
    print(f"   - Bias Factor: {parcel_developed_pct/developed_pct:.2f}x overestimation by parcel count")
    
    # Find dominant land use by acres
    dominant_use = max(stats['land_use_acres'].items(), key=lambda x: x[1])
    print(f"\n2. DOMINANT LAND USE:")
    print(f"   - {dominant_use[0]}: {dominant_use[1]:,.1f} acres ({stats['landscape_percentages'][dominant_use[0]]:.2f}%)")
    
    # Calculate development density
    if stats['total_parcels'] > 0:
        developed_parcels = df[df['developed_pct'] > 50]
        print(f"\n3. DEVELOPMENT PATTERNS:")
        print(f"   - Parcels >50% Developed: {len(developed_parcels):,} ({len(developed_parcels)/stats['total_parcels']*100:.1f}%)")
        print(f"   - Average Size of Developed Parcels: {developed_parcels['CAL_ACREAGE'].mean():.2f} acres")
        print(f"   - Average Size of All Parcels: {df['CAL_ACREAGE'].mean():.2f} acres")
    
    # Save summary to file
    summary_file = Path('outputs') / 'acre_weighted_analysis.txt'
    with open(summary_file, 'w') as f:
        f.write("ACRE-WEIGHTED LAND USE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Parcels: {stats['total_parcels']:,}\n")
        f.write(f"Total Acres: {stats['total_acres']:,.1f}\n\n")
        
        f.write("Landscape Composition (Acre-Weighted):\n")
        for land_use, pct in stats['landscape_percentages'].items():
            f.write(f"  {land_use}: {pct:.2f}% ({stats['land_use_acres'][land_use]:,.1f} acres)\n")
        
        f.write(f"\nKey Finding: {developed_pct:.2f}% of the landscape acres are developed\n")
        f.write(f"(compared to {parcel_developed_pct:.2f}% when measured by parcel count)\n")
    
    print(f"\nAnalysis saved to: {summary_file}")

if __name__ == "__main__":
    main()