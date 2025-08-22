#!/bin/bash

# Run the parcel land use zonal statistics pipeline

echo "================================================"
echo "PARCEL LAND USE ZONAL STATISTICS PIPELINE"
echo "================================================"
echo ""

# Parse command line arguments
SAMPLE_SIZE=""
CHUNK_SIZE="5000"
RESUME=""
DRY_RUN=""
METHOD="subpixel"

while [[ $# -gt 0 ]]; do
  case $1 in
    --sample)
      SAMPLE_SIZE="--sample $2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --resume)
      RESUME="--resume"
      shift
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run_pipeline.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --sample N       Process only N parcels (for testing)"
      echo "  --chunk-size N   Number of parcels per chunk (default: 5000)"
      echo "  --resume         Resume from checkpoint"
      echo "  --dry-run        Analyze data without processing"
      echo "  --method METHOD  Zonal stats method: subpixel (default, adaptive), standard, center"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Run the pipeline
echo "Starting pipeline..."
echo "Chunk size: $CHUNK_SIZE"
echo "Method: $METHOD ($([ "$METHOD" = "subpixel" ] && echo "adaptive resolution, 99% accurate" || echo "testing only"))"

if [ -n "$SAMPLE_SIZE" ]; then
  echo "Sample mode: $SAMPLE_SIZE parcels"
fi

if [ -n "$RESUME" ]; then
  echo "Resume mode: Continuing from checkpoint"
fi

if [ -n "$DRY_RUN" ]; then
  echo "Dry run mode: Analysis only"
fi

echo ""
echo "Running pipeline..."
echo "----------------------------------------"

uv run python -m src.main \
  --chunk-size "$CHUNK_SIZE" \
  --method "$METHOD" \
  $SAMPLE_SIZE \
  $RESUME \
  $DRY_RUN

echo ""
echo "Pipeline complete!"