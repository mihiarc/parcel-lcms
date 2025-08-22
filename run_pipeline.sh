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
PARALLEL=""

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
    --parallel)
      PARALLEL="--parallel"
      shift
      ;;
    --no-parallel)
      PARALLEL="--no-parallel"
      shift
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
      echo "  --parallel       Enable parallel processing (default: enabled)"
      echo "  --no-parallel    Disable parallel processing"
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

# Display parallel processing mode
if [ "$PARALLEL" = "--no-parallel" ]; then
  echo "Processing mode: Sequential (single core)"
elif [ "$PARALLEL" = "--parallel" ] || [ -z "$PARALLEL" ]; then
  echo "Processing mode: Parallel (${N_WORKERS:-12} workers)"
fi

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
  $PARALLEL \
  $SAMPLE_SIZE \
  $RESUME \
  $DRY_RUN

echo ""
echo "Pipeline complete!"