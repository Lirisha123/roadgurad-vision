
import os
import gradio as gr
import spaces
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"

CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    background: #080d1b !important;
}
body, .gradio-container {
    background: #080d1b !important;
    color: #e8efff !important;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #142b55, #201642, #102d38);
    border: 1px solid #34466e;
    margin-bottom: 20px;
}
.hero h1 { font-size: 38px; margin-bottom: 8px; color: #ffffff; }
.hero p { color: #bdcbea; font-size: 15px; }
.metric {
    padding: 18px;
    border-radius: 14px;
    background: #111e35;
    border: 1px solid #2d4268;
}
.metric h2 { color: #6ee7f9; margin: 0; }
.metric p { color: #b7c5df; margin-bottom: 0; }
.panel {
    padding: 20px;
    border: 1px solid #283653;
    border-radius: 16px;
    background: #10182a;
}
footer { display: none !important; }
"""

def image_if_exists(path):
    return path if os.path.isfile(path) else None


@spaces.GPU(duration=30)
def detect_image(image, confidence):
    if image is None:
        return None, "Upload a road image to start detection."

    if not os.path.isfile(MODEL_PATH):
        return None, "Error: `best.pt` was not found beside app.py."

    try:
        model = YOLO(MODEL_PATH)
        results = model.predict(
            source=image,
            conf=float(confidence),
            imgsz=640,
            device=0,
            verbose=False
        )

        result = results[0]
        annotated = Image.fromarray(
            result.plot()[..., ::-1].copy()
        )

        counts = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = result.names[class_id]
            counts[label] = counts.get(label, 0) + 1

        total = sum(counts.values())
        lines = [
            "## Detection complete",
            f"**Objects detected:** {total}",
            f"**Confidence threshold:** {float(confidence):.2f}",
            "",
            "### Detected classes"
        ]

        if counts:
            for label, count in counts.items():
                lines.append(f"- **{label}:** {count}")
        else:
            lines.append(
                "No objects detected above the selected confidence threshold."
            )

        return annotated, "\n".join(lines)

    except Exception as error:
        return None, (
            "### Detection failed\n"
            f"`{type(error).__name__}: {error}`\n\n"
            "Check the Space logs for more details."
        )


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="indigo",
        neutral_hue="slate"
    ),
    css=CSS,
    title="RoadGuard Vision Lab"
) as demo:

    gr.HTML("""
    <div class="hero">
      <p>ROADGUARD AI · COMPUTER VISION LAB</p>
      <h1>RoadGuard Vision Lab</h1>
      <p>AI-powered pavement distress detection for safer roads.
      Upload a road image and inspect the model's predictions.</p>
    </div>
    """)

    with gr.Tabs():

        # DETECTION LAB
        with gr.Tab("🔎 Detection Lab"):
            gr.Markdown("## Inspect a road image")
            gr.Markdown(
                "Upload a road image to detect cracks and potholes."
            )

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload road image",
                        type="pil"
                    )
                    confidence = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.25,
                        step=0.05,
                        label="Confidence threshold"
                    )
                    run_button = gr.Button(
                        "Run RoadGuard Detection",
                        variant="primary"
                    )

                with gr.Column():
                    output_image = gr.Image(
                        label="Detection result",
                        type="pil"
                    )
                    output_report = gr.Markdown(
                        "Your detection summary will appear here."
                    )

            run_button.click(
                fn=detect_image,
                inputs=[input_image, confidence],
                outputs=[output_image, output_report]
            )

        # OVERVIEW
        with gr.Tab("📊 Overview"):
            gr.Markdown("## Model overview")
            gr.Markdown(
                "RoadGuard uses YOLOv11n to identify pavement distress "
                "in two classes: cracks and potholes."
            )

            with gr.Row():
                gr.HTML("""
                <div class="metric">
                    <h2>0.828</h2>
                    <p>Validation mAP@50</p>
                </div>
                """)
                gr.HTML("""
                <div class="metric">
                    <h2>0.520</h2>
                    <p>Validation mAP@50–95</p>
                </div>
                """)
                gr.HTML("""
                <div class="metric">
                    <h2>0.845</h2>
                    <p>Validation precision</p>
                </div>
                """)
                gr.HTML("""
                <div class="metric">
                    <h2>0.736</h2>
                    <p>Validation recall</p>
                </div>
                """)

            gr.Markdown(
                "These are reported validation metrics, not measurements "
                "from every uploaded image or the separate field-test set."
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Training results")
                    path = image_if_exists("results.png")
                    if path:
                        gr.Image(
                            value=path,
                            label="Training curves",
                            interactive=False
                        )
                    else:
                        gr.Markdown("`results.png` was not found.")

                with gr.Column():
                    gr.Markdown("### Confusion matrix")
                    path = image_if_exists("confusion_matrix.png")
                    if path:
                        gr.Image(
                            value=path,
                            label="Validation confusion matrix",
                            interactive=False
                        )
                    else:
                        gr.Markdown("`confusion_matrix.png` was not found.")

        # FIELD TESTING
        with gr.Tab("🛣️ Field Testing"):
            gr.Markdown("## Road image examples")
            gr.Markdown(
                "Saved detection examples from the project."
            )

            with gr.Row():
                for path, label in [
                    ("detection.png", "Detection example 1"),
                    ("detection2.png", "Detection example 2")
                ]:
                    if os.path.isfile(path):
                        with gr.Column():
                            gr.Image(
                                value=path,
                                label=label,
                                interactive=False
                            )
                    else:
                        gr.Markdown(f"`{path}` was not found.")

            gr.HTML("""
            <div class="panel">
                <h3>Held-out field testing</h3>
                <p>183 real smartphone road photos were collected
                separately for field testing. This set was not used
                to train the model.</p>
            </div>
            """)

        # MY CONTRIBUTION / NOVELTIES
        with gr.Tab("🚀 My Novelties"):
            gr.Markdown("## Model development & novelties")
            gr.Markdown(
                "Practical improvements focused on road-image variability, "
                "background interference, and detection performance."
            )

            with gr.Row():
                gr.HTML("""
                <div class="panel">
                    <h3>01 · Hard-negative mining</h3>
                    <p>Included background images with empty labels to help
                    the detector learn from non-distress scenes and reduce
                    false detections on ordinary road surfaces.</p>
                </div>
                """)
                gr.HTML("""
                <div class="panel">
                    <h3>02 · Targeted augmentation</h3>
                    <p>Tuned image augmentation to improve robustness to
                    changing road-image conditions.</p>
                    <p><b>HSV value:</b> 0.4<br>
                    <b>HSV saturation:</b> 0.5<br>
                    <b>Rotation:</b> 15°<br>
                    <b>Horizontal flip:</b> 0.5</p>
                </div>
                """)

            with gr.Row():
                gr.HTML("""
                <div class="panel">
                    <h3>03 · Validation-led improvement</h3>
                    <p>Compared model performance during development and
                    reported an increase in validation mAP@50 from
                    <b>0.564 to 0.828</b>.</p>
                    <p>This is a validation result, not a guarantee of
                    equivalent performance in every real-world setting.</p>
                </div>
                """)
                gr.HTML("""
                <div class="panel">
                    <h3>04 · Lightweight deployment</h3>
                    <p>Used the YOLOv11n model for a compact inference
                    pipeline suitable for an interactive demonstration.</p>
                    <p><b>Model size:</b> approximately 5.5 MB<br>
                    <b>Parameters:</b> approximately 2.58 million</p>
                </div>
                """)

            gr.Markdown("### Training configuration")
            gr.Markdown("""
            - **Architecture:** YOLOv11n
            - **Detection classes:** Crack and pothole
            - **Training duration:** 50 epochs
            - **Batch size:** 16
            - **Image size:** 640 × 640
            - **Training hardware:** Tesla T4
            """)

demo.launch()
