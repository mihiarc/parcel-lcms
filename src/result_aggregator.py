"""Result aggregation and output utilities."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

import pandas as pd
import geopandas as gpd

from .config import (
    OUTPUT_DIR,
    OUTPUT_FORMAT,
    COMPRESSION,
    OUTPUT_COLUMNS,
    PARCEL_ID_FIELD,
    LAND_USE_CLASSES
)

logger = logging.getLogger(__name__)

class ResultAggregator:
    """Aggregate and save processing results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize result aggregator.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = {}
        
    def aggregate_results(
        self,
        results_list: list,
        parcels_gdf: Optional[gpd.GeoDataFrame] = None
    ) -> gpd.GeoDataFrame:
        """Aggregate results from multiple chunks.
        
        Args:
            results_list: List of result DataFrames
            parcels_gdf: Original parcels with geometry
            
        Returns:
            Aggregated GeoDataFrame
        """
        logger.info(f"Aggregating {len(results_list)} result chunks")
        
        # Concatenate results
        if isinstance(results_list[0], pd.DataFrame):
            results_df = pd.concat(results_list, ignore_index=True)
        else:
            results_df = results_list[0]  # Single result
        
        logger.info(f"Aggregated {len(results_df)} parcels")
        
        # Add geometry if provided
        if parcels_gdf is not None:
            logger.info("Joining results with original geometries")
            
            # Merge on parcel ID
            results_gdf = parcels_gdf[[PARCEL_ID_FIELD, 'geometry']].merge(
                results_df,
                on=PARCEL_ID_FIELD,
                how='inner'
            )
            
            # Convert to GeoDataFrame
            results_gdf = gpd.GeoDataFrame(results_gdf, geometry='geometry')
        else:
            results_gdf = gpd.GeoDataFrame(results_df)
        
        # Reorder columns
        available_cols = [col for col in OUTPUT_COLUMNS if col in results_gdf.columns]
        other_cols = [col for col in results_gdf.columns if col not in available_cols]
        results_gdf = results_gdf[available_cols + other_cols]
        
        return results_gdf
    
    def generate_summary_statistics(self, results_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Generate summary statistics from results.
        
        Args:
            results_gdf: Results GeoDataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary statistics")
        
        summary = {
            'total_parcels': len(results_gdf),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Land use distribution
        if 'majority_land_use' in results_gdf.columns:
            land_use_counts = results_gdf['majority_land_use'].value_counts()
            summary['land_use_distribution'] = land_use_counts.to_dict()
            
            # Calculate total area by land use
            proportion_cols = [
                'agriculture_pct', 'developed_pct', 'forest_pct',
                'other_pct', 'rangeland_pasture_pct'
            ]
            
            if all(col in results_gdf.columns for col in proportion_cols):
                avg_proportions = results_gdf[proportion_cols].mean()
                summary['average_proportions'] = avg_proportions.to_dict()
        
        # Pixel statistics
        if 'valid_pixels' in results_gdf.columns:
            summary['pixel_statistics'] = {
                'total_valid_pixels': results_gdf['valid_pixels'].sum(),
                'avg_pixels_per_parcel': results_gdf['valid_pixels'].mean(),
                'parcels_with_no_pixels': (results_gdf['valid_pixels'] == 0).sum()
            }
        
        # Validation statistics
        if 'is_valid' in results_gdf.columns:
            summary['validation'] = {
                'valid_parcels': results_gdf['is_valid'].sum(),
                'invalid_parcels': (~results_gdf['is_valid']).sum(),
                'validation_rate': results_gdf['is_valid'].mean() * 100
            }
        
        logger.info(f"Summary: {summary}")
        
        return summary
    
    def save_results(
        self,
        results_gdf: gpd.GeoDataFrame,
        output_name: str = "parcel_land_use_results",
        format: str = OUTPUT_FORMAT
    ) -> Path:
        """Save results to file.
        
        Args:
            results_gdf: Results GeoDataFrame
            output_name: Base name for output file
            format: Output format ('geoparquet', 'geojson', 'shapefile', 'csv')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "geoparquet":
            output_file = self.output_dir / f"{output_name}_{timestamp}.parquet"
            results_gdf.to_parquet(output_file, compression=COMPRESSION)
            
        elif format == "geojson":
            output_file = self.output_dir / f"{output_name}_{timestamp}.geojson"
            results_gdf.to_file(output_file, driver='GeoJSON')
            
        elif format == "shapefile":
            output_file = self.output_dir / f"{output_name}_{timestamp}"
            output_file.mkdir(exist_ok=True)
            results_gdf.to_file(output_file / f"{output_name}.shp")
            
        elif format == "csv":
            output_file = self.output_dir / f"{output_name}_{timestamp}.csv"
            # Drop geometry for CSV
            results_df = pd.DataFrame(results_gdf.drop(columns='geometry', errors='ignore'))
            results_df.to_csv(output_file, index=False)
            
        else:
            raise ValueError(f"Unknown output format: {format}")
        
        logger.info(f"Saved results to {output_file}")
        
        return output_file
    
    def save_summary(self, summary: Dict[str, Any], output_name: str = "processing_summary") -> Path:
        """Save processing summary to JSON.
        
        Args:
            summary: Summary dictionary
            output_name: Output file name
            
        Returns:
            Path to saved summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"{output_name}_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary to {summary_file}")
        
        return summary_file
    
    def create_report(
        self,
        results_gdf: gpd.GeoDataFrame,
        processing_stats: Dict[str, Any]
    ) -> str:
        """Create a text report of processing results.
        
        Args:
            results_gdf: Results GeoDataFrame
            processing_stats: Processing statistics
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 60,
            "PARCEL LAND USE ZONAL STATISTICS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PROCESSING SUMMARY",
            "-" * 40,
            f"Total parcels processed: {len(results_gdf):,}",
        ]
        
        # Add processing time if available
        if 'total_time_seconds' in processing_stats:
            total_time = processing_stats['total_time_seconds']
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            report_lines.append(f"Processing time: {hours}h {minutes}m {seconds}s")
            
            if 'average_parcels_per_second' in processing_stats:
                pps = processing_stats['average_parcels_per_second']
                report_lines.append(f"Average speed: {pps:.1f} parcels/second")
        
        report_lines.extend(["", "LAND USE DISTRIBUTION", "-" * 40])
        
        # Land use statistics
        if 'majority_land_use' in results_gdf.columns:
            land_use_counts = results_gdf['majority_land_use'].value_counts()
            total = land_use_counts.sum()
            
            for land_use, count in land_use_counts.items():
                percentage = (count / total) * 100
                report_lines.append(f"{land_use:20} {count:8,} ({percentage:5.1f}%)")
        
        # Average proportions
        proportion_cols = [
            'agriculture_pct', 'developed_pct', 'forest_pct',
            'other_pct', 'rangeland_pasture_pct'
        ]
        
        if all(col in results_gdf.columns for col in proportion_cols):
            report_lines.extend(["", "AVERAGE LAND USE PROPORTIONS", "-" * 40])
            
            for col in proportion_cols:
                avg_pct = results_gdf[col].mean()
                land_use_name = col.replace('_pct', '').replace('_', ' ').title()
                report_lines.append(f"{land_use_name:20} {avg_pct:6.2f}%")
        
        # Validation statistics
        if 'is_valid' in results_gdf.columns:
            report_lines.extend(["", "VALIDATION RESULTS", "-" * 40])
            
            n_valid = results_gdf['is_valid'].sum()
            n_invalid = (~results_gdf['is_valid']).sum()
            validation_rate = (n_valid / len(results_gdf)) * 100
            
            report_lines.extend([
                f"Valid parcels:   {n_valid:8,} ({validation_rate:.1f}%)",
                f"Invalid parcels: {n_invalid:8,} ({100-validation_rate:.1f}%)"
            ])
        
        report_lines.extend(["", "=" * 60])
        
        report = "\n".join(report_lines)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"processing_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved report to {report_file}")
        
        return report