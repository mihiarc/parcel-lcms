#!/usr/bin/env python
"""Create a sample of 1000 parcels from Benton County and save to CSV."""

import pandas as pd
import geopandas as gpd
from pathlib import Path

def main():
    # Load processed parcel data
    parquet_file = Path('outputs/parcel_land_use_results_20250822_075345.parquet')
    print(f"Loading processed parcel data: {parquet_file}")
    
    # Load full dataset
    df = pd.read_parquet(parquet_file)
    print(f"Total rows loaded: {len(df):,}")
    
    # Remove duplicates first
    df_unique = df.drop_duplicates(subset=['PARCEL_LID'], keep='first')
    print(f"Unique parcels: {len(df_unique):,}")
    
    # Load county boundaries to identify Benton County parcels
    county_shapefile = Path('data/tl_2024_us_county/tl_2024_us_county.shp')
    if county_shapefile.exists():
        print(f"\nLoading county boundaries: {county_shapefile}")
        counties = gpd.read_file(county_shapefile)
        
        # Filter for Benton County, Oregon (STATEFP = '41' for Oregon)
        benton = counties[(counties['STATEFP'] == '41') & (counties['NAME'] == 'Benton')]
        
        if not benton.empty:
            print(f"Found Benton County, Oregon")
            
            # Convert parcels to GeoDataFrame
            print("\nConverting to GeoDataFrame for spatial filtering...")
            gdf = gpd.read_parquet(parquet_file)
            
            # Remove duplicates
            gdf_unique = gdf.drop_duplicates(subset=['PARCEL_LID'], keep='first')
            
            # Ensure both are in the same CRS
            if gdf_unique.crs != benton.crs:
                print(f"Reprojecting Benton County from {benton.crs} to {gdf_unique.crs}")
                benton = benton.to_crs(gdf_unique.crs)
            
            # Spatial join to find parcels in Benton County
            print("Finding parcels within Benton County...")
            benton_parcels = gpd.sjoin(gdf_unique, benton, how='inner', predicate='intersects')
            
            print(f"Found {len(benton_parcels):,} parcels in Benton County")
            
            # Sample 1000 parcels (or all if less than 1000)
            sample_size = min(1000, len(benton_parcels))
            benton_sample = benton_parcels.sample(n=sample_size, random_state=42)
            
            # Prepare for CSV output (drop geometry and county columns)
            columns_to_keep = [
                'PARCEL_LID', 'CAL_ACREAGE',
                'agriculture_pct', 'developed_pct', 'forest_pct', 
                'other_pct', 'rangeland_pasture_pct',
                'majority_land_use', 'total_pixels', 'valid_pixels'
            ]
            
            # Add optional columns if they exist
            optional_cols = ['acre_diff', 'acre_diff_pct', 'calculated_acres', 'subpixel_resolution']
            for col in optional_cols:
                if col in benton_sample.columns:
                    columns_to_keep.append(col)
            
            # Filter to only existing columns
            columns_to_keep = [col for col in columns_to_keep if col in benton_sample.columns]
            
            csv_data = benton_sample[columns_to_keep].copy()
            
            # Sort by PARCEL_LID for consistency
            csv_data = csv_data.sort_values('PARCEL_LID')
            
            # Save to CSV
            output_file = Path('outputs/benton_county_sample_1000.csv')
            csv_data.to_csv(output_file, index=False)
            
            print(f"\n{'='*60}")
            print("BENTON COUNTY SAMPLE CREATED")
            print('='*60)
            print(f"Sample size: {len(csv_data):,} parcels")
            print(f"Output file: {output_file}")
            
            # Print summary statistics
            print("\nSample Statistics:")
            print(f"  Total acres: {csv_data['CAL_ACREAGE'].sum():,.1f}")
            print(f"  Average parcel size: {csv_data['CAL_ACREAGE'].mean():.2f} acres")
            print(f"  Median parcel size: {csv_data['CAL_ACREAGE'].median():.2f} acres")
            
            print("\nLand Use Distribution (averages):")
            print(f"  Agriculture: {csv_data['agriculture_pct'].mean():.2f}%")
            print(f"  Developed: {csv_data['developed_pct'].mean():.2f}%")
            print(f"  Forest: {csv_data['forest_pct'].mean():.2f}%")
            print(f"  Other: {csv_data['other_pct'].mean():.2f}%")
            print(f"  Rangeland/Pasture: {csv_data['rangeland_pasture_pct'].mean():.2f}%")
            
            print("\nMajority Land Use Counts:")
            majority_counts = csv_data['majority_land_use'].value_counts()
            for land_use, count in majority_counts.items():
                print(f"  {land_use}: {count} parcels ({count/len(csv_data)*100:.1f}%)")
            
            return csv_data
            
        else:
            print("Benton County not found in county shapefile")
    else:
        print(f"County shapefile not found: {county_shapefile}")
        print("\nFalling back to PARCEL_LID pattern matching...")
        
        # Alternative: Try to identify Benton County parcels by ID pattern
        # Oregon counties often have codes - Benton County might have a specific prefix
        # This is less reliable but worth trying
        
        # Sample based on general distribution
        print("Creating a general sample of 1000 Oregon parcels instead...")
        
        sample = df_unique.sample(n=min(1000, len(df_unique)), random_state=42)
        
        # Prepare for CSV
        columns_to_keep = [
            'PARCEL_LID', 'CAL_ACREAGE',
            'agriculture_pct', 'developed_pct', 'forest_pct', 
            'other_pct', 'rangeland_pasture_pct',
            'majority_land_use', 'total_pixels', 'valid_pixels'
        ]
        
        # Filter to only existing columns
        columns_to_keep = [col for col in columns_to_keep if col in sample.columns]
        csv_data = sample[columns_to_keep].copy()
        csv_data = csv_data.sort_values('PARCEL_LID')
        
        output_file = Path('outputs/oregon_sample_1000.csv')
        csv_data.to_csv(output_file, index=False)
        
        print(f"\nCreated general Oregon sample: {output_file}")
        print(f"Sample size: {len(csv_data):,} parcels")

if __name__ == "__main__":
    main()