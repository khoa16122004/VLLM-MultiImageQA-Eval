# VLLM-MultiImage-RAG-VQA-Eval
This repo contains the inference, evaluation code, retrieval and Python environment file for LVLMs that can process multiple images as input across various multi-image visual question-answering datasets.

# Usage
Move to the repo directory to the folder ```src```

## 🔍 Step 1: Extract Features & Create Vector Database
- For extract feature and create vector database run the folowing script:
```bash
    CUDA_VISIBLE_DEVICES=<id> python create_db.py \
    --model_name_encode CLIP \
    --dataset_dir ../dataset/MRAG_corpus \
    --database_dir new_database/MRAG/CLIP \
    --output_index_dir new_database/MRAG/CLIP/index
```

## 📁 Step 2: Extract Retrieval Paths

- For extract the json containing the path results when retriev the question img, uisng the following script:

```bash
    CUDA_VISIBLE_DEVICES=3 python extract_retrieval.py \
    --model_name_encode CLIP \
    --index_dir ../new_database/MRAG/CLIP/index
```


<!-- # [MSRAG BenchMark](https://github.com/mragbench/MRAG-Bench)  
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

</p> -->

