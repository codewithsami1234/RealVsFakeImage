import os

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


# Get the directory where this app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directory
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "ai_vs_real_image_detection"
)

print(f"Loading model from: {MODEL_DIR}")

# Load image processor
processor = ViTImageProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

# Load model
model = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.eval()

print("Model loaded successfully!")
print("Labels:", model.config.id2label)


# ZeroGPU function
@spaces.GPU
def predict(image):
    if image is None:
        return "No image provided", 0.0

    # Make sure image is a PIL image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convert to RGB
    image = image.convert("RGB")

    # Process image
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # Get predicted class
    predicted_class_id = probabilities.argmax().item()

    predicted_label = model.config.id2label[
        predicted_class_id
    ]

    # Get confidence
    confidence = (
        probabilities[predicted_class_id].item()
        * 100
    )

    # Convert labels to user-friendly text
    if predicted_label.upper() == "FAKE":
        display_label = "AI Generated Image"

    elif predicted_label.upper() == "REAL":
        display_label = "Real Image"

    else:
        display_label = predicted_label

    print(
        f"Prediction: {display_label} | "
        f"Confidence: {confidence:.2f}%"
    )

    return display_label, round(confidence, 2)


# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),
    outputs=[
        gr.Textbox(
            label="Prediction"
        ),
        gr.Number(
            label="Confidence (%)"
        )
    ],
    title="AI vs Real Image Detector",
    description=(
        "Upload an image to determine whether "
        "it is AI-generated or real."
    )
)

demo.launch()