
INPUT_DIR="/path/to/images"
PARQUET_DIR="/path/to/parquet/files"
OUTPUT_DIR="/path/to/output/files"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Call generate.py with the provided arguments
python generate.py \
    --input_dir "$INPUT_DIR" \
    --parquet_dir "$PARQUET_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Data generation complete. Results saved to $OUTPUT_DIR"
