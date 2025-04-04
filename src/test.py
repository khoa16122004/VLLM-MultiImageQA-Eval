import os

dataset_dir = "../dataset/MRAG"
count = 0
for sample_id in os.listdir(dataset_dir):
    if sample_id.endswith(".py") or sample_id == "index":
        continue
    sample_dir_input = os.path.join(dataset_dir, sample_id)
    for file_name in os.listdir(sample_dir_input):
        if "gt" in file_name:
            count += 1
            
print(count)