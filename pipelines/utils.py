import torch
import daft
from PIL import Image
import open_clip
import numpy as np

from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

@daft.udf(return_dtype=daft.DataType.tensor(daft.DataType.float32(), shape=(3, 224, 224)))
class NormalizeCLIPImageUDF:
    MODEL_NAME = "ViT-H-14-quickgelu"
    MODEL_BASE = "metaclip_fullcc"

    def __init__(self):
        _, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
    
    def _process_image(self, series):
        data = [Image.fromarray(image.squeeze() if image.shape[-1] == 1 else image).convert("RGB") for image in series.to_pylist()]

        return [self.preprocess(sample) for sample in data]

    def __call__(self, series):
        return self._process_image(series)

@daft.udf(return_dtype=daft.DataType.tensor(daft.DataType.float32(), shape=(3, 384, 384)))
class NormalizeSIGLIPImageUDF:
    MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
    MODEL_BASE = "webli"

    def __init__(self):
        _, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
    
    def _process_image(self, series):
        data = [Image.fromarray(image.squeeze() if image.shape[-1] == 1 else image).convert("RGB") for image in series.to_pylist()]

        return [self.preprocess(sample) for sample in data]

    def __call__(self, series):
        return self._process_image(series)
    

@daft.udf(return_dtype=daft.DataType.embedding(daft.DataType.float32(), 1024))
class CLIPTextUDF:
    DEVICE = "cuda"
    BATCH_SIZE = 524288
    MODEL_NAME = "ViT-H-14-quickgelu"
    MODEL_BASE = "metaclip_fullcc"

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _process_text(self, series, scalar):
        data = self.tokenizer(series.to_pylist())

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), self.BATCH_SIZE):
                batch = data[i:i + self.BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_text(batch)
                normalized_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                # Multiply the normalized features by the scalar
                multiplied_features = normalized_features * scalar

                features.extend(torch.unbind(multiplied_features.cpu()))

        return features

    def __call__(self, series, scalar=1):
        return self._process_text(series, scalar)


@daft.udf(return_dtype=daft.DataType.embedding(daft.DataType.float32(), 1152))
class SIGLIPTextUDF:
    DEVICE = "cuda"
    BATCH_SIZE = 1024
    MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
    MODEL_BASE = "webli"

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _process_text(self, series, scalar):
        data = self.tokenizer(series.to_pylist())

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), self.BATCH_SIZE):
                batch = data[i:i + self.BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_text(batch)
                normalized_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                # Multiply the normalized features by the scalar
                multiplied_features = normalized_features * scalar

                features.extend(torch.unbind(multiplied_features.cpu()))

        return features

    def __call__(self, series, scalar=1):
        return self._process_text(series, scalar)

@daft.udf(return_dtype=daft.DataType.embedding(daft.DataType.float32(), 1024))
class CLIPImageUDF:
    DEVICE = "cuda"
    BATCH_SIZE = 1024
    MODEL_NAME = "ViT-H-14-quickgelu"
    MODEL_BASE = "metaclip_fullcc"

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _process_image(self, series):
        data = torch.tensor(np.array(series.to_pylist()))

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), self.BATCH_SIZE):
                batch = data[i:i + self.BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_image(batch)

                features.extend(torch.unbind((batch_features / batch_features.norm(dim=-1, keepdim=True)).cpu()))

        return features

    def __call__(self, series):
        return self._process_image(series)
    

@daft.udf(return_dtype=daft.DataType.embedding(daft.DataType.float32(), 1152))
class SIGLIPImageUDF:
    DEVICE = "cuda"
    BATCH_SIZE = 1024
    MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
    MODEL_BASE = "webli"

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _process_image(self, series):
        data = torch.tensor(np.array(series.to_pylist()))

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), self.BATCH_SIZE):
                batch = data[i:i + self.BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_image(batch)

                features.extend(torch.unbind((batch_features / batch_features.norm(dim=-1, keepdim=True)).cpu()))

        return features

    def __call__(self, series):
        return self._process_image(series)
    

@daft.udf(return_dtype=daft.DataType.embedding(daft.DataType.float32(), 1024))
class CLIPUDF:
    DEVICE = "cuda"
    MODEL_NAME = "ViT-H-14-quickgelu"
    MODEL_BASE = "metaclip_fullcc"

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _process_image(self, series):
        BATCH_SIZE = 1024
        data = torch.tensor(np.array(series.to_pylist()))

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_image(batch)

                features.extend(torch.unbind((batch_features / batch_features.norm(dim=-1, keepdim=True)).cpu()))

        return features
    
    def _process_text(self, series, scalar):
        BATCH_SIZE = 524288
        data = self.tokenizer(series.to_pylist())

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_text(batch)
                normalized_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                # Multiply the normalized features by the scalar
                multiplied_features = normalized_features * scalar

                features.extend(torch.unbind(multiplied_features.cpu()))

        return features

    def __call__(self, image, text, scalar):
        return self._process_image(image) + self._process_text(text, scalar)
    

@daft.udf(return_dtype=daft.DataType.embedding(daft.DataType.float32(), 1152))
class SIGLIPUDF:
    DEVICE = "cuda"
    MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
    MODEL_BASE = "webli"

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.MODEL_BASE)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _process_image(self, series):
        BATCH_SIZE = 1024
        data = torch.tensor(np.array(series.to_pylist()))

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_image(batch)

                features.extend(torch.unbind((batch_features / batch_features.norm(dim=-1, keepdim=True)).cpu()))

        return features
    
    def _process_text(self, series, scalar):
        BATCH_SIZE = 524288
        data = self.tokenizer(series.to_pylist())

        features = []

        with torch.no_grad(), torch.autocast(self.DEVICE):
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE].to(self.DEVICE)

                batch_features = self.model.encode_text(batch)
                normalized_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                # Multiply the normalized features by the scalar
                multiplied_features = normalized_features * scalar

                features.extend(torch.unbind(multiplied_features.cpu()))

        return features

    def __call__(self, image, text, scalar):
        return self._process_image(image) + self._process_image(text, scalar)


def greedy(c1, c2):
    """
    Match clusters from two lists using a greedy algorithm.

    Args:
    c1 (list of list): First list of clusters.
    c2 (list of list): Second list of clusters.

    Returns:
    list of tuple: List of matched cluster pairs.
    """
    # Sort clusters by size in decreasing order
    c1 = sorted(c1, key=len, reverse=True)
    c2 = sorted(c2, key=len, reverse=True)

    matched_pairs = []
    num_samples = 0

    while c1 and c2:
        # Pick the larger of the first element of c1 or c2
        if len(c1[0]) >= len(c2[0]):
            larger_cluster = c1.pop(0)
            smaller_list = c2
        else:
            larger_cluster = c2.pop(0)
            smaller_list = c1

        # Find the best match for the larger cluster in the smaller list
        best_match = None
        best_score = -1
        for i, cluster in enumerate(smaller_list):
            overlap = len(set(larger_cluster) & set(cluster))
            avg_size = (len(larger_cluster) + len(cluster)) / 2
            score = overlap / avg_size
            if score > best_score:
                best_score = score
                best_match = (i, cluster)

        if best_match:
            best_index, best_cluster = best_match
            union = list(set(larger_cluster) | set(best_cluster))
            matched_pairs.append(union)
            num_samples += len(union)
            
            del smaller_list[best_index]

    return matched_pairs, num_samples


def cluster(labels, images):
    """
    Construct clusters of lists of lists, where each sublist contains all valid images corresponding to a specific label.

    Parameters:
    image_urls (list): The list of image names.
    labels (numpy.ndarray): The array of cluster labels, where -1 indicates noise.

    Returns:
    list: A list of lists, where each sublist contains images corresponding to a specific cluster label.
    """

    # Create a dictionary to hold lists of images for each label
    label_to_images = {}

    # Iterate over each valid label and corresponding image URL
    for label, image in zip(labels, images):
        label_to_images[label] = label_to_images.get(label, [])
        label_to_images[label].append(image)

    # Convert the dictionary to a list of lists
    list_of_clusters = list(label_to_images.values())

    return list_of_clusters

import os
from together import Together

def generate(captions):
    client = Together()
    instruction = """You are an AI visual assistant that can analyze multiple images. You receive four to five images with their corresponding captions in an array. The task is to use the provided images and captions to create a challenging and complex question that involves comparison, ranking, storytelling, or logical reasoning across the images and then provide a detailed answer. The question should require a deep understanding of the visual content and advanced reasoning based on that content. Create questions that ask to compare elements across the images, such as identifying which image best represents a critical or turning point moment, quality, or characteristic; formulate questions that require ranking the images based on intricate and plausible criteria, such as strategic importance, sequence, or visual impact; develop questions that involve piecing together a narrative from the images, understanding a sophisticated sequence of events, or explaining a complex progression shown; and ask questions that require advanced logical reasoning to deduce why certain elements are present, the purpose behind actions shown, or the broader implications of what is depicted. Example questions include: “Which image best represents the pivotal turning point of the event, and why?” “Rank the images based on the strategic importance of the actions shown, from highest to lowest.” “How do the images collectively tell the intricate story of the event, and what can be inferred about the key strategic moments?” and “What could be the underlying reasons behind the specific actions taken in each image, and how do they relate to the overall context in a broader sense?” Frame a question that requires advanced analyzing and reasoning about the images and their captions, and provide a detailed answer based on the visual content and captions, explaining the reasoning process and the conclusions drawn. Return your challenging and complex question in the format: "Question: [INSERT QUESTION], Answer: [INSERT ANSWER]:"""

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": "".join(f"Image {i+1} Caption:" + captions[i] for i in range(len(captions))) + instruction}],
    )

    return response.choices[0].message.content

import re
# Function to format generated question answer pairs into dictionary
def get_qa(text):
  text = text.replace('\n', ' ')
  question_match = re.search(r'Question:\s*(.*?)\s*Answer:', text, re.DOTALL)
  answer_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)

  if question_match and answer_match:
    question = question_match.group(1).strip()
    answer = answer_match.group(1).strip()
    
  return question, answer