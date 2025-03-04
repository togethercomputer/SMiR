ENDPOINT_URL=""
PROFILE_NAME=""
BUCKET_NAME=""
DATA_PATH=""
IMAGES_PATH=""
OUTPUT_PATH=""
SAMPLE_SIZE=20000
C_VALUE=0.2

# Ensure the output directory exists
mkdir -p "$OUTPUT_PATH"

# Call embed.py with the provided arguments
python embed.py \
    --endpoint-url "$ENDPOINT_URL" \
    --profile-name "$PROFILE_NAME" \
    --bucket-name "$BUCKET_NAME" \
    --data-path "$DATA_PATH" \
    --images-path "$IMAGES_PATH" \
    --output-path "$OUTPUT_PATH" \
    --sample-size "$SAMPLE_SIZE" \
    --c "$C_VALUE"

echo "Embedding process completed. Results saved to $OUTPUT_PATH"