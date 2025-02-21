# AI-Image-Captioning
An **AI-powered image captioning app** built with **Streamlit**, using **ViT-GPT2** for caption generation and **YOLOv8** for object detection. The app enhances captions by integrating detected objects into the generated text.

## 🔥 Features
- **AI-powered image captioning** using **ViT-GPT2**.
- **Object detection** with **YOLOv8** to enhance captions.
- **Dark-themed UI** with **Streamlit**.
- **Interactive settings** for enabling/disabling object detection.
- **Optimized inference** with GPU acceleration (CUDA support).

## 🚀 Demo
### 1️⃣ **Upload an Image**
![Upload Screenshot](https://github.com/PrachiPatel15/AI-Image-Captioning/blob/main/assest/upload_image.png)

### 2️⃣ **Enable Object Detection and Generate Captions**
![Detection Screenshot](https://github.com/PrachiPatel15/AI-Image-Captioning/blob/main/assest/object_detection_tick.png)

### 3️⃣ **View Enhanced Caption and Detected Objects**
![Results Screenshot](https://github.com/PrachiPatel15/AI-Image-Captioning/blob/main/assest/obj_with_caption.png)

## 📂 Installation & Setup
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/AI-Image-Captioning.git
cd AI-Image-Captioning
```

### 2️⃣ **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Application**
```bash
streamlit run app.py
```

## 🧠 Models Used
### **1️⃣ ViT-GPT2** (Image Captioning)
- **Pretrained Model**: `nlpconnect/vit-gpt2-image-captioning`
- **Task**: Generates textual descriptions for input images.

### **2️⃣ YOLOv8** (Object Detection)
- **Pretrained Model**: `yolov8n.pt`
- **Task**: Detects objects in the image to enhance captions.

## ⚙️ Project Structure
```bash
AI-Image-Captioning/
│── app.py                  # Main Streamlit application
│── requirements.txt        # Required dependencies
│── README.md               # Documentation
│── assets/                 # Store images/screenshots
```

## 🛠️ Usage Instructions
1. **Upload an image** in the app.
2. **Choose whether to enable object detection**.
3. **Click 'Analyze Image'** to generate a caption.
4. **View enhanced captions** and object detection results.

## 💡 Future Improvements
- [ ] Add multilingual captioning support.
- [ ] Optimize object detection performance.
- [ ] Implement additional caption refinement techniques.

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** this repository and create a **pull request** with your improvements.

