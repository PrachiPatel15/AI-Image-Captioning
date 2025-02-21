import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np
from ultralytics import YOLO

# Load models
@st.cache_resource
def load_models():
    # Load pre-trained ViT-GPT2 model for image captioning
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Load YOLOv8 model
    detection_model = YOLO("yolov8n.pt") 
    
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_model.to(device)
    
    return caption_model, feature_extractor, tokenizer, detection_model, device

# Generate basic caption
def predict_caption(image, model, feature_extractor, tokenizer, device):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
        
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    max_length, num_beams = 30, 5
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

# Detect objects in the image
def detect_objects(image, model):
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    results = model(img_array)
    
    detected_objects, confidence_scores = [], {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            conf = float(box.conf[0].item())
            
            if label not in detected_objects:
                detected_objects.append(label)
                confidence_scores[label] = conf
            elif conf > confidence_scores[label]:
                confidence_scores[label] = conf
    
    plot = results[0].plot()
    detected_image = Image.fromarray(plot)
    return detected_objects, detected_image, confidence_scores

# Enhance caption based on detected objects
def enhance_caption_advanced(basic_caption, detected_objects, confidence_scores=None):
    if not detected_objects:
        return basic_caption
    
    # Scene categorization logic (same as before)
    outdoor_objects = {'tree', 'car', 'person', 'dog', 'bicycle', 'bird', 'mountain', 'sky'}
    indoor_objects = {'chair', 'table', 'sofa', 'tv', 'book', 'bottle', 'cup', 'bed'}
    food_objects = {'pizza', 'sandwich', 'apple', 'banana', 'cake', 'donut', 'bowl'}
    tech_objects = {'laptop', 'cell phone', 'keyboard', 'mouse', 'tv', 'remote'}
    
    outdoor_count = sum(1 for obj in detected_objects if obj.lower() in outdoor_objects)
    indoor_count = sum(1 for obj in detected_objects if obj.lower() in indoor_objects)
    food_count = sum(1 for obj in detected_objects if obj.lower() in food_objects)
    tech_count = sum(1 for obj in detected_objects if obj.lower() in tech_objects)
    
    max_count = max(outdoor_count, indoor_count, food_count, tech_count)
    scene_context = None
    if max_count > 0:
        if outdoor_count == max_count:
            scene_context = "outdoor"
        elif indoor_count == max_count:
            scene_context = "indoor"
        elif food_count == max_count:
            scene_context = "food"
        elif tech_count == max_count:
            scene_context = "technology"
    
    object_counts = {obj: detected_objects.count(obj) for obj in set(detected_objects)}
    sorted_objects = sorted(object_counts.keys(), key=lambda obj: confidence_scores.get(obj, 0), reverse=True)
    
    mentioned_objects = [obj for obj in detected_objects if obj.lower() in basic_caption.lower()]
    unmentioned = [obj for obj in sorted_objects if obj not in mentioned_objects]
    
    if scene_context == "outdoor":
        return f"{basic_caption} This outdoor scene features {', '.join(unmentioned)}."
    elif scene_context == "indoor":
        return f"{basic_caption} Inside this space, you can see {', '.join(unmentioned)}."
    elif scene_context == "food":
        return f"{basic_caption} The meal includes {', '.join(unmentioned)}."
    elif scene_context == "technology":
        return f"{basic_caption} This setup includes {', '.join(unmentioned)}."
    
    return basic_caption

# Main function
def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced Image Captioning",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for full-width layout and styling
    st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f9f9f9;
        }
        .stApp {
            max-width: 100%;
            margin: auto;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #333333;
        }
        .stButton button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stImage img {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-container {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            gap: 20px;
        }
        .column {
            flex: 1;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("📷 Advanced Image Captioning")
    st.write("Upload an image to generate AI-powered captions with optional object detection.")

    # # Sidebar
    # st.sidebar.title("⚙️ Settings")
    # st.sidebar.info("Customize your experience:")
    # detection_enabled = st.sidebar.checkbox("Enable Object Detection", value=True)

    with st.sidebar:
        st.sidebar.title("⚙️ Settings")
        detection_enabled = st.toggle("Enable object detection", value=True)
        
        st.markdown("### About")
        st.markdown("""
        This app combines Vision Transformer with YOLOv8 object 
        detection to create detailed image descriptions.
        
        **Models used:**
        - ViT-GPT2 for base captioning
        - YOLOv8n for object detection
        """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

    # Load models
    try:
        caption_model, feature_extractor, tokenizer, detection_model, device = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        detection_enabled = False
        caption_model, feature_extractor, tokenizer, device = load_models()

    if uploaded_file:
        image = Image.open(uploaded_file)

        if detection_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    basic_caption = predict_caption(image, caption_model, feature_extractor, tokenizer, device)
                    detected_objects, detected_image, confidence_scores = detect_objects(image, detection_model)
                    enhanced_caption = enhance_caption_advanced(basic_caption, detected_objects, confidence_scores)
                
                with col2:
                    st.subheader("Object Detection")
                    st.image(detected_image, caption="Detected Objects", use_container_width=True)
                
                st.subheader("Generated Captions")
                st.write(f"**Basic Caption:** {basic_caption}")
                st.write(f"**Enhanced Caption:** {enhanced_caption}")
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    caption = predict_caption(image, caption_model, feature_extractor, tokenizer, device)
                st.subheader("Generated Caption")
                st.write(f"**{caption}**")

if __name__ == "__main__":
    main()