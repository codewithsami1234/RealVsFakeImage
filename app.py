import os

import gradio as gr
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "ai_vs_real_image_detection"
)

print(f"Loading model from: {MODEL_DIR}")


processor = ViTImageProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.eval()

print("Model loaded successfully!")
print("Labels:", model.config.id2label)


def predict(image):
    if image is None:
        return "No image provided", 0.0

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_class_id = probabilities.argmax().item()

    predicted_label = model.config.id2label[
        predicted_class_id
    ]

    confidence = (
        probabilities[predicted_class_id].item()
        * 100
    )

    if predicted_label.upper() == "FAKE":
        display_label = "AI Generated Image"

    elif predicted_label.upper() == "REAL":
        display_label = "Real Image"

    else:
        display_label = predicted_label

    return display_label, round(confidence, 2)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence (%)")
    ],
    title="AI vs Real Image Detector",
    description=(
        "Upload an image to determine whether "
        "it is AI-generated or real."
    )
)

demo.launch()
