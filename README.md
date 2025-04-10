# VLLM-MultiImage-RAG-VQA-Eval
This repo contains the inference, evaluation code, retrieval and Python environment file for LVLMs that can process multiple images as input across various multi-image visual question-answering datasets.


# Usage
Move to the direct direcotry to the folder ```src```
For extract feature and create vector database run the folowing script:
```cmd
    
```


# [MSRAG BenchMark](https://github.com/mragbench/MRAG-Bench)  
This dataset contains the benchmark for multi-image retrived for visual questioning with multiple-choice questions.

## Accuracy Table

<p align="center">

| Model                          | Without GT Image       | With 5 GT Images       | With 5 retrieval image (50 reranking) (CLIP)
|--------------------------------|:----------------------:|:----------------------:| :----------------------:
| **Mantis-8B-clip-llama3**      | 41.09                  | 44.05                  | 
| **Mantis-8B-sigclip-llama3**   | 43.9                   | 49.0                   |
| **Deepseek-VL-7B-chat**        | 43.75                  | 50.03                  |
| **LLaVA-NeXT-Interleave-7B**   | 44.20                  | 54.18                  | 
| **LLaVA-OneVision**            | 53.07                  | 60.46                  | 47.8

</p>

