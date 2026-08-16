# 🌿 Plant Disease Identification using CNN

A deep learning-based web application that detects plant diseases from leaf images using a **Convolutional Neural Network (CNN)**. Built with TensorFlow/Keras and deployed with a Streamlit web interface.

---

## 📌 Project Overview

Early detection of plant diseases is critical for reducing crop loss and minimizing pesticide use. This system allows farmers and agronomists to simply upload a photo of a plant leaf and instantly receive:

- ✅ Disease diagnosis (Healthy / Diseased)
- 🦠 Disease name and description
- 📊 AI confidence score
- ⚠️ Severity level
- 🌱 Organic treatment recommendations
- 🧪 Chemical treatment recommendations
- 🛡️ Prevention tips

---

## 🖥️ Screenshots

| Leaf Scanner | Disease Library |
|---|---|
| Upload leaf → instant AI diagnosis | Browse 38 disease classes |

---

## 🚀 Features

- 🔬 **Leaf Scanner** — Upload any leaf photo and get instant disease detection
- 📚 **Disease Library** — Browse all 38 plant diseases with detailed info
- 📊 **Session Dashboard** — Track your scanning history and statistics
- ⚙️ **Model & Training** — View CNN architecture and retrain on custom datasets
- 🎛️ **Image Enhancement** — Adjust brightness, contrast, and sharpness before analysis
- 🕓 **Scan History** — Last 10 scans saved per session

---

## 🧠 Model Architecture

```
Input Image (256 × 256 × 3)
    ↓
Conv2D(32) + ReLU + BatchNorm + MaxPool(3×3) + Dropout(0.25)
    ↓
Conv2D(64) + ReLU + BatchNorm
Conv2D(64) + ReLU + BatchNorm + MaxPool(2×2) + Dropout(0.25)
    ↓
Conv2D(128) + ReLU + BatchNorm
Conv2D(128) + ReLU + BatchNorm + MaxPool(2×2) + Dropout(0.25)
    ↓
Flatten → Dense(512) + ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense(38) + Softmax → Disease Classification
```

- **Total Parameters:** ~29.2 Million
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (lr = 0.001)
- **Target Accuracy:** ~95% on PlantVillage test set

---

## 🌱 Dataset

- **Name:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Images:** 54,000+ leaf images
- **Classes:** 38 (diseases + healthy plants)
- **Plants Covered:** Apple, Corn (Maize), Grape, Potato, Tomato, Pepper (Bell)

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Programming Language |
| TensorFlow 2.x / Keras | Deep Learning Model |
| OpenCV | Image Processing |
| Pillow (PIL) | Image Enhancement |
| Streamlit | Web Interface |
| Matplotlib | Training Plots & Charts |
| Scikit-learn | Label Encoding |
| NumPy | Numerical Operations |

---

## 📂 Project Structure

```
Plant-Disease-Identification-using-CNN/
├── app.py                    # Main Streamlit web application
├── PlantDiseaseDetection.py  # CNN model training script
├── create_samples.py         # Sample image generator
├── plant_disease_model.h5    # Pre-trained Keras model
├── cnn_model.pkl             # Pre-trained model (pickle)
├── label_transform.pkl       # Label binarizer (pickle)
├── training_plot.png         # Accuracy & loss curves
├── sample_images/            # Sample leaf images for testing
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ShreyaVinayakSawant/last-year-enginerring-project.git
cd last-year-enginerring-project
```

### 2. Install Dependencies
```bash
pip install streamlit tensorflow opencv-python pillow matplotlib scikit-learn numpy
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## 🎯 How to Use

1. Open the app at `http://localhost:8501`
2. Navigate to **🔬 Leaf Scanner** in the sidebar
3. Upload a leaf photo (JPG/PNG) or select a sample image
4. Results appear automatically:
   - Disease name & status
   - AI confidence score
   - Severity level
   - Treatment & prevention tips

---

## 🌾 Supported Diseases (38 Classes)

| Plant | Diseases |
|---|---|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |

---

## 📈 Training Your Own Model

1. Download [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle
2. Extract to `./plantvillage/` folder
3. Run training:
```bash
python PlantDiseaseDetection.py
```
4. Or use the **⚙️ Model & Training** page in the web app

---

## 👩‍💻 Author

**Shreya Vinayak Sawant**  
Final Year Engineering Project  
Plant Disease Detection using Deep Learning (CNN)

---

## 📄 License

This project is for educational purposes.  
Dataset credit: [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
