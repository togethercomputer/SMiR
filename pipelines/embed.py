import os
import argparse
import boto3
import numpy as np
import daft
import functools

from utils import NormalizeSIGLIPImageUDF, SIGLIPTextUDF, SIGLIPImageUDF

def embed(endpoint_url, profile_name, bucket_name, data_path, images_path, output_path, sample_size=20000, c=0.2):
    # Set up Daft runner
    daft.context.set_runner_ray()
    
    # Configure S3 access
    BUCKET_PREFIX = "s3://"
    BUCKET_URL = f"{BUCKET_PREFIX}{bucket_name}"
    
    s3_config = daft.io.S3Config(endpoint_url=endpoint_url, profile_name=profile_name) 
    io_config = daft.io.IOConfig(s3=s3_config)
    
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3", endpoint_url=endpoint_url)
    
    # List files in the data path
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=data_path)
    s3_files = [content["Key"] for content in response.get("Contents")]
    
    # Read and process data
    dfs = [daft.io.read_parquet(os.path.join(BUCKET_URL, file), io_config=io_config) for file in s3_files]
    df = functools.reduce(lambda x, y: x.concat(y), dfs)
    df = df.where(~df["image_url"].is_null() & (df["image_url"].str.rstrip() != ""))
    
    # Determine number of partitions
    num_partitions = int(np.ceil(df.count_rows() / sample_size))
    df = df.into_partitions(num_partitions)
    
    df.explain(show_all=True)
    
    # Download images based on url
    df = df.with_column(
        "data",
        (daft.lit(os.path.join(BUCKET_URL, images_path, "")) + daft.col("image_url")).url.download(max_connections=256, io_config=io_config),
    )
    
    # Decode images
    df = df.where(~daft.col('data').is_null())
    
    df = df.with_column(
        "image",
        df["data"].image.decode())
    df = df.where(~daft.col('image').is_null())
    
    # Generate embeddings
    df = df.with_column(
        "siglip_normalized_image",
        NormalizeSIGLIPImageUDF(df["image"]),
    )
    
    df = df.with_column(
        "siglip_text",
        SIGLIPTextUDF(df["text"], c),
        resource_request=daft.ResourceRequest(num_gpus=1),
    )
    
    df = df.with_column(
        "siglip_image",
        SIGLIPImageUDF(df["siglip_normalized_image"]),
        resource_request=daft.ResourceRequest(num_gpus=1),
    )
    
    df = df.with_column(
        "siglip",
        df["siglip_text"] + df["siglip_image"]
    )
    
    print('calculated embeddings')
    
    df = df.select(daft.col('image_url'), daft.col('text'), daft.col('siglip'))
    
    df.write_parquet(root_dir=output_path)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate SIGLIP embeddings for image-text pairs')
    parser.add_argument('--endpoint-url', required=True, help='S3 endpoint URL')
    parser.add_argument('--profile-name', required=True, help='AWS profile name')
    parser.add_argument('--bucket-name', required=True, help='S3 bucket name')
    parser.add_argument('--data-path', required=True, help='Path to data in S3 bucket')
    parser.add_argument('--images-path', required=True, help='Path to images in S3 bucket')
    parser.add_argument('--output-path', required=True, help='Local path to save embeddings')
    parser.add_argument('--sample-size', type=int, default=20000, help='Sample size for partitioning')
    parser.add_argument('--c', type=float, default=0.2, help='Parameter for SIGLIP text embedding')
    args = parser.parse_args()
    
    # Call the embed function with parsed arguments
    embed(
        args.endpoint_url,
        args.profile_name,
        args.bucket_name,
        args.data_path,
        args.images_path,
        args.output_path,
        args.sample_size,
        args.c
    )
