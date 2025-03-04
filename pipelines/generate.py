# Generate based on softmax graph
import os
import json
import ray
import pandas as pd
import numpy as np
import cupy as cp
from cuml.preprocessing import normalize
from tqdm import tqdm
import random
import argparse

from utils import generate, get_qa

ray.init()

@ray.remote(num_gpus=1)
def process_parquet(parquet_file, input_dir, output_dir):

    data_list = []
    
    df = pd.read_parquet(parquet_file)

    X = normalize(cp.array(np.vstack(df['siglip'])))

    def softmax(x):
        e_x = np.exp(x - np.max(x))  # For numerical stability
        return e_x / e_x.sum(axis=0)

    def random_sample_with_softmax(X, num_samples):
        # Pick a row at random
        n_rows = X.shape[0]
        random_index = np.random.randint(n_rows)

        # Compute the Euclidean distances
        selected_embedding = X[random_index]
        distances = np.linalg.norm(X - selected_embedding, axis=1)

        # To avoid selecting the same embedding, set its distance to infinity
        distances[random_index] = np.inf

        # Convert distances to probabilities using softmax
        probabilities = softmax(-distances)  # Negate distances because closer embeddings should have higher probabilities

        # Sample indices of the remaining embeddings based on the probabilities
        sampled_indices = np.random.choice(np.arange(n_rows), size=num_samples, replace=False, p=probabilities.get())

        # Combine the selected embedding index with the sampled indices into a list
        indices_list = [random_index] + sampled_indices.tolist()

        return indices_list

    # Randomly sample elements without replacement and proportionally

    numbers = [4, 5]
    weights = [0.35, 0.65]
    dataset = 1000

    for i in tqdm(range(dataset)):
        num_samples = random.choices(numbers, weights=weights, k=1)[0]

        sampled_indices = random_sample_with_softmax(X, num_samples)
        samples = df.loc[sampled_indices]
        sampled_elements = samples['image_url']
        
        captions = []
        for url in sampled_elements:
            row = samples[samples['image_url'] == url]

            caption = row['text'].tolist()[0]
            captions.append(caption)

        question, answer = get_qa(generate(captions))

        if question and answer:

            question += "<image>" * len(sampled_elements)
            
            entry = {
                "id": f'graph_{i}',
                "images": [os.path.join(input_dir, url) for url in sampled_elements],
                "conversation": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                "source": "synthetic_from_graph"
            }
            data_list.append(entry)

    
    # Create output filename
    base_name = os.path.basename(parquet_file)
    output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.json")
    
    with open(output_file, 'w') as f:
        json.dump(json.loads(data_list), f)
    
    return f"Completed processing {parquet_file}"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process parquet files to generate synthetic data')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--parquet_dir', required=True, help='Directory containing parquet files')
    parser.add_argument('--output_dir', required=True, help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    parquet_dir = args.parquet_dir
    output_dir = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all parquet files
    parquet_files = [
        os.path.join(parquet_dir, f) 
        for f in os.listdir(parquet_dir) 
        if f.endswith('.parquet')
    ]

    # Submit tasks to Ray
    futures = [process_parquet.remote(file, input_dir, output_dir) for file in parquet_files]
    
    # Wait for all tasks to complete and print results
    for result in ray.get(futures):
        print(result)