# 🧠 Image Classification Web App (Cats, Dogs & Flowers)

This project classifies images of cats, dogs, and five flower types (daisy, dandelion, rose, sunflower, and tulip) using a VGG16 model and Flask web interface.

---

## 🚀 Features

- Cat vs Dog classification
- Flower type recognition
- Real-time predictions via web UI

---

## 🛠️ Tech Stack

- Python
- Flask
- TensorFlow / Keras (VGG16)
- NumPy
- Matplotlib

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MHS-007/AI-Lab-Project.git
cd AI-Lab-Project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. Run the Flask app

```bash
python app.py
```

Server will start at: http://127.0.0.1:5000

---

## 📂 Dataset Structure & Setup

⚠️ Note: The complete `datasets/` folder is **not included** in this repository due to large file size. So, first You must download both datasets and then create the `datasets/` folder manually by following the suggested structure given below in **Folder Layout** section. Then add this folder with the rest of the folder structure (after cloning the repo) before running the training scripts.

### 🧾 Folder Layout (Expected Structure)

datasets/  
│
├── cats_and_dogs/  
│ ├── PetImages/  
│ │ ├── Cat/ → 0.jpg ... 12498.jpg  
│ │ └── Dog/ → 0.jpg ... 12498.jpg  
│ ├── Train/  
│ │ ├── train_cats/ → 0.jpg ... 4165.jpg  
│ │ └── train_dogs/ → 0.jpg ... 4165.jpg  
│ ├── Valid/  
│ │ ├── valid_cats/ → 8333.jpg ... 12498.jpg  
│ │ └── valid_dogs/ → 8333.jpg ... 12498.jpg  
│ ├── Test/  
│ │ ├── test_cats/ → 4166.jpg ... 8332.jpg  
│ │ └── test_dogs/ → 4166.jpg ... 8332.jpg  
│ └── NewData/ → small set (6 images) for real-time prediction.  
│  
└── flowers/  
├── train/  
│ ├── daisy/ → 501 Images  
│ ├── tulip/ → 607  
│ ├── sunflower/ → 495 Images  
│ ├── rose/ → 497 Images  
│ └── dandelion/ → 646 Images  
├── test/  
│ └── test_flowers/ → 924 total images across all classes  
└── new_data/ → 5 sample images for final testing

- All images from `PetImages` folder were **manually split** into training, validation, and testing sets to ensure balanced distribution.
- `NewData` and `new_data` folders contain small sets of unseen images used for **actual prediction testing** in Flask. You can download the random images of cats, dogs and flowers (5 Categories) for testing as well.

💡 Tip: Once you recreate the structure, ensure your local folder names match exactly as shown above, otherwise Keras generators will fail to locate directories.

---

## 📊 Model Training

Model was trained on the following datasets:
- **Cats & Dogs dataset** sourced from [Kaggle](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)
- **Flowers dataset** sourced from [Kaggle](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)

---

## Models

Download trained models from:
[Google Drive](https://drive.google.com/drive/folders/1t5kNfNIgCBTpGDWRbzrhra81HeSLVhJj?usp=sharing)

---

## 💡 How It Works

&nbsp; 1. User uploads an image through the web UI.  
&nbsp; 2. Flask backend loads the trained VGG16 model.  
&nbsp; 3. The image is preprocessed (resized, normalized).  
&nbsp; 4. Model predicts the class with confidence score.  
&nbsp; 5. Result is displayed on the web page instantly.

---

## 📸 Demo
Here’s a sample output from the web app:

![App Screenshot](https://github.com/MHS-007/AI-Lab-Project/blob/450e32a262f1638e65a5c69c3a3b57274f3862c0/Screenshots/Screenshot_01.png)

---

## ⚠️ Notes & Limitations

- The models were trained on limited datasets; predictions may vary with unseen or poor-quality images.
- Flask app runs locally by default; online deployment may require additional setup (e.g., Render, Hugging Face, or Streamlit).
- Model files (.h5) are large and hosted externally on Google Drive. Ensure you download them before running predictions.
- Current version supports only cats, dogs, and five flower types (daisy, tulip, sunflower, rose, dandelion).

---

## 👨‍💻 Author  
This project was developed as part of my course **AI Lab** at _Iqra University_.
