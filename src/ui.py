import gradio as gr
import os
from PIL import Image

def load_sample(sample_id, retrivels_names):
    question_dir = "dataset/MRAG"
    dataset_dir = "dataset/mrag_bench_image_corpus\image_corpus"
    folder_path = os.path.join(question_dir, sample_id)    
    if not os.path.exists(question_dir):
        return "Không tìm thấy thư mục", None, []

    question = "Không tìm thấy câu hỏi."
    question_img = None
    gt_files = []
    retrivels_files = [Image.open(os.path.join(dataset_dir, file_name)).convert("RGB") for file_name in retrivels_names]

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.png') and "gt" in filename:
            gt_files.append(Image.open(file_path).convert("RGB"))

        elif filename.endswith('.png') and "question" in filename:
            question_img = Image.open(file_path).convert("RGB")

        elif filename.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                question = f.read()

    return question, question_img, gt_files, retrivels_files

demo = gr.Interface(
    fn=load_sample,
    inputs=gr.Textbox(label="Nhập Sample ID (VD: 0, 1, 2...)"),
    outputs=[
        gr.Textbox(label="Câu hỏi"),
        gr.Image(label="question_img", scale=0.5),
        gr.Gallery(label="Ground Truth Images", columns=5, height=200),
        gr.Gallery(label="Retrieved Images", columns=5, height=200)
    ],
    title="MRAG Sample Viewer",
    description="Nhập sample ID để xem ảnh và câu hỏi tương ứng."
)

if __name__ == "__main__":
    demo.launch()
