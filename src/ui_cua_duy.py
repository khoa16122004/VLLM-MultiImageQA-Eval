from datasets import load_dataset
import gradio as gr
import os
from PIL import Image
import re
import json

mrag_bench = load_dataset("uclanlp/MRAG-Bench", split="test")

id_to_index = {int(item["id"]): idx for idx, item in enumerate(mrag_bench)}

def load_sample(sample_id, retrievals_json):
    dataset_dir = os.path.join(r"C:\Users\hokha\OneDrive\Desktop\workplaces\captioning\mrag_bench_image_corpus\image_corpus")

    sample = mrag_bench[id_to_index[int(sample_id)]]

    question_img = sample["image"]
    question = f'{sample["question"]}\nA: {sample["A"]}\nB: {sample["B"]}\nC: {sample["C"]}\nD: {sample["D"]}\nAnswer: {sample["answer"]}'
    gt_images = [iamge for iamge in sample["gt_images"]]
    with open(retrievals_json, "r", encoding="utf-8") as f:
        retrievals = json.load(f)
    retrievals_images = [Image.open(os.path.join(dataset_dir, image)).convert("RGB") for image in retrievals[str(sample['id'])]]
    retrievals_path = [image.replace('_', ' ').replace('-', ' ') for image in retrievals[str(sample['id'])]]

    return question, question_img, gt_images, retrievals_images, retrievals_path
    

demo = gr.Interface(
    fn=load_sample,
    inputs=[
        gr.Textbox(label="Nhập Sample ID (VD: 0, 1, 2...)"),
        gr.Textbox(label="json chứa ảnh retrieved (VD: {0: ['img1.png', 'img2.png']})")
    ],
    outputs=[
        gr.Textbox(label="Câu hỏi"),
        gr.Image(label="question_img", scale=0.5),
        gr.Gallery(label="Ground Truth Images", columns=5, height=200),
        gr.Gallery(label="Retrieved Images", columns=5, height=200),
        gr.Textbox(label="retrievals_path")
    ],
    title="MRAG Sample Viewer",
    description="Nhập sample ID và tên ảnh retrieved để xem ảnh và câu hỏi tương ứng."
)

if __name__ == "__main__":
    demo.launch()
