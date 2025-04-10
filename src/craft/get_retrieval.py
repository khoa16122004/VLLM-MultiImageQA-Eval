import json
import os

def get_retrieval(index, retrieval_path="../database/ReT_retrieval/mrag_bench_top5_results.json"):
    if not os.path.exists(retrieval_path):
        raise FileNotFoundError(f"Không tìm thấy file: {retrieval_path}")
    
    with open(retrieval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(index, int):
        index = str(index)
    
    if index not in data:
        raise KeyError(f"Không tìm thấy chỉ mục: {index}")
    
    retrieval_data = data[index]
    return retrieval_data

if __name__ == "__main__":
    try:
        index = 500
        retrieval_data = get_retrieval(index)
        print(retrieval_data)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")