# 🐾 Pet Image Classifier with CNN in PyTorch

This project trains a Convolutional Neural Network (CNN) using PyTorch to classify pet images—specifically cats vs. dogs—based on image data. It uses a custom image dataset structured for binary classification and evaluates model performance with accuracy, precision, and recall.

---

## 🧠 Model Architecture

- Conv2d(3, 16, kernel_size=3, padding=1) → ReLU → MaxPool2d
- Conv2d(16, 32, kernel_size=3, padding=1) → ReLU → MaxPool2d
- Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool2d
- Flatten → Linear(64×12×12 → 128) → ReLU
- Linear(128 → 64) → ReLU
- Linear(64 → 2) → Output logits for 2 classes

---

## 🖼️ Dataset Structure

The model uses images loaded via `torchvision.datasets.ImageFolder` with the following folder layout:

./petimages/
├── cat/
│ ├── image1.jpg
│ └── ...
├── dog/
│ ├── image1.jpg
│ └── ...

yaml
Copy
Edit

---

## 🔄 Preprocessing

- Resize to 100×100
- Convert to PyTorch tensor
- Normalize with mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

---

## ⚙️ Training Details

- Loss: CrossEntropyLoss  
- Optimizer: Adam (lr = 0.0001)  
- Epochs: 10  
- Batch Size: 32  
- Uses CUDA if available

---

## 📊 Evaluation Metrics

The model evaluates performance using:

- **Accuracy**
- **Recall** (macro average)
- **Precision** (macro average)

These are printed after training using `sklearn.metrics`.

---

## 📈 Example Output

[10, 100] loss: 0.512
Finished Training

Accuracy: 0.87
Recall: 0.86
Precision: 0.87

yaml
Copy
Edit

---

## 🚀 Future Enhancements

- Add data augmentation for robustness
- Use transfer learning (e.g., ResNet18)
- Deploy with Streamlit or Flask for live testing

---
