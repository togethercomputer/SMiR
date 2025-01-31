# SMiR
Synthetic data pipeline for multi-image reasoning

## Overview
This repository contains the official implementation of our paper: [Efficient Synthetic Data Pipeline to Improve Multi-Image Reasoning](https://arxiv.org/abs/2501.03675).

## Coming Soon
- Dataset generation pipeline


## 🏆 Credits

We would like to acknowledge the following resources that were instrumental in the development of SMIR:

- [Meta Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct): We utilized the Llama 3.1 model as our foundational language model via ["Together AI"](https://www.together.ai/models/llama-3-1-70b).

- [SigLIP](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384: We utilized a SigLIP model as our embedding model from Google.

- [CLIP](https://github.com/facebookresearch/MetaCLIP/blob/main/src/open_clip/model_configs/ViT-H-14-quickgelu.json): We utilized MetaCLIP, Meta's implementation of CLIP, as our embedding model.

- We used training and evaluation code from the following repositories:
  - [MANTIS: Interleaved Multi-Image Instruction Tuning](https://github.com/TIGER-AI-Lab/Mantis)
  - [From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline](https://github.com/lmarena/arena-hard-auto)

<a name="bibtex"/>

## 📚 BibTeX

```bibtex
@misc{li2025smirefficientsyntheticdata,
      title={SMIR: Efficient Synthetic Data Pipeline To Improve Multi-Image Reasoning}, 
      author={Andrew Li and Rahul Thapa and Rahul Chalamala and Qingyang Wu and Kezhen Chen and James Zou},
      year={2025},
      eprint={2501.03675},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.03675}, 
}
```