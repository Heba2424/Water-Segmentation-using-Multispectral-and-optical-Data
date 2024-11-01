# Water-Segmentation-using-Multispectral-and-optical-Data

# Image Segmentation with U-Net

This project implements an image segmentation model using U-Net architecture with a ResNet34 encoder. The model is trained to segment images stored in TIFF format and masks stored in PNG format. The implementation leverages PyTorch and the `segmentation-models-pytorch` library for model building and training.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [License](#license)

## Features
- Preprocessing of images and masks
- Binary segmentation using U-Net architecture
- Evaluation metrics: Loss, Dice score, F1 score, and Accuracy
- Flask deployment for serving the model

## Requirements
Make sure you have the following libraries installed:
- Python 3.x
- PyTorch
- segmentation-models-pytorch
- torchmetrics
- rasterio
- tifffile
- torchvision
- matplotlib
- scikit-learn
- Flask

## Dataset
Ensure you have your dataset organized as follows:
- Place your `.tif` images in the `data/images/` directory.
- Place your corresponding `.png` masks in the `data/labels/` directory.

## Installation
To run this project, you'll need to install the required packages. You can do this using pip:

```bash
pip install -r requirements.txt
```

## Usage
*Instructions for usage go here.*

## Training
To train the model, run the following command in your terminal:
```bash
python src/train.py
```

This will initiate the training process, using the images and masks you prepared earlier. The training loop will print the training and validation metrics, including loss, Dice score, F1 score, and accuracy.

## Evaluation
To evaluate the trained model on a test set, ensure your test dataset is similarly organized as the training data and run:
```bash
python src/evaluate.py
```
This will output the test loss, Dice score, F1 score, and accuracy.

## Results
Upon evaluation, you can expect output metrics similar to the following:
```yaml
Test Loss: 0.6287
Test Dice Score: 0.8970
Test F1 Score: 0.8970
Test Accuracy: 0.9530
```
![App Result](app%20result.png)
## Deployment
To deploy the model as a web application using Flask, follow these steps:

1. **Install Flask:**
   ```bash
   pip install Flask
   ```
2. **Create a app.py file in the root of your project with the following content:**
   ```python
   from flask import Flask, request, jsonify
   import torch
   from torchvision import transforms
   import segmentation_models_pytorch as smp
   from PIL import Image
   import numpy as np

   app = Flask(__name__)

   # Load your trained model
   model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
   )
   model.load_state_dict(torch.load('path_to_your_trained_model.pth'))
   model.eval()

   # Define the image transformations
   transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the input image to the model's expected input size
    transforms.ToTensor(),            # Convert the image to a tensor
   ])

   @app.route('/predict', methods=['POST'])
   def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Process the input image
    img = Image.open(file.stream).convert("RGB")  # Read the image and convert to RGB
    img = transform(img)  # Apply the transformations
    img = img.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(img)
        prediction = torch.sigmoid(output).squeeze().numpy()  # Apply sigmoid and convert to numpy array

    # Convert the prediction to a binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8)  # Threshold the output

    return jsonify({'prediction': binary_mask.tolist()})  # Return the prediction as a list

   if __name__ == '__main__':
    app.run(debug=True)

   ```
 3. **Run the Flask app: In your terminal, run**
    ```bash
     python app.py
     ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- **U-Net:** Convolutional Networks for Biomedical Image Segmentation
- **Segmentation Models PyTorch:** A library that provides implementations of segmentation models in PyTorch.
