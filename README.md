# PyTorch_food_vision

**PyTorch Model Deployment: FoodVision Big** 🍔👁

**Overview**

This project demonstrates an end-to-end workflow for deploying a deep learning model built using PyTorch. The goal is to classify images of food into 101 different categories using the EfficientNetB2 architecture. The model is deployed using Gradio, providing a user-friendly interface to make predictions on uploaded images. This project covers the entire process, from model training to deployment in a production environment.

The model used in this project is based on EfficientNetB2, a powerful convolutional neural network architecture optimized for both performance and efficiency. The model has been trained to classify food images into one of 101 different categories, using the Food-101 dataset. Pre-trained weights were used to fine-tune the model for this specific task.

**Installation**

To get started with this project, you need to clone the repository and install the required dependencies.

# Clone the repository
```
git clone https://github.com/your-username/pytorch-model-deployment.git
```

# Change into the directory
```
cd pytorch-model-deployment
```

# Create and activate a virtual environment (optional but recommended)
```
python3 -m venv env
source env/bin/activate
```

# Install the required dependencies
```
pip install -r requirements.txt
```

**Usage**

Run the notebook : Open the 09_pytorch_model_deployment.ipynb file in Jupyter Notebook or Jupyter Lab and execute the cells to see the model in action.
Deploy with Gradio: To interact with the model via the Gradio interface, simply run the following command from your terminal or notebook:
bash
```
python app.py
```
This will launch a Gradio web interface where you can upload an image of food and receive predictions from the model.

Predict Function: A custom prediction function is used to preprocess images and perform classification. The function returns the prediction and the time taken for inference.
```
def predict(img) -> Tuple[Dict, float]:
    # Image transformation and inference code goes here
    return pred_labels_and_probs, pred_time
```
    
**Deployment**

The project leverages Gradio to create an interactive web-based interface where users can upload food images and see real-time predictions. Gradio is a great tool for demoing machine learning models and allows for easy deployment without complex web development.

**Results:**

The model achieves high accuracy in classifying food images, and predictions are made in under a second. The EfficientNetB2 model's fine-tuned version is efficient and performs well on a variety of food types.

