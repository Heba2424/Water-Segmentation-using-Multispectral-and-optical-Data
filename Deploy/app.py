from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import tifffile as tiff
from PIL import Image
import os
import base64
from io import BytesIO

# Initialize the Flask app
app = Flask(__name__)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and pre_conv layer
pre_conv = nn.Conv2d(12, 3, kernel_size=1)
model = smp.Unet(
    encoder_name="resnet34",  # Encoder: resnet34
    encoder_weights=None,     # No pretrained weights, load from trained weights
    in_channels=3,            # Input channels after the 1x1 convolution
    classes=1,                # Output classes (1 for binary segmentation)
    activation=None           # No activation in the final layer
)

# Load the saved model state and pre_conv state
model_data = torch.load('D:/Skills/Traning/Cellula Tec/Stellite Water Segmentation/Deploy/unet_model.pth', map_location=device)

# Load the state dicts into the layers
pre_conv.load_state_dict(model_data['pre_conv_state_dict'])
model.load_state_dict(model_data['model_state_dict'])

# Transfer model and pre_conv to the device
pre_conv.to(device)
model.to(device)

# Forward pass function
def forward(x):
    x = pre_conv(x)  # Apply 1x1 convolution to reduce the input channels
    return model(x)

# Helper function to preprocess the input image
def preprocess_image(image_path):
    # Use tifffile to handle .tif files
    if image_path.lower().endswith('.tif'):
        with tiff.TiffFile(image_path) as tif:
            image = tif.asarray()
        
        # Check the shape of the image
        if len(image.shape) == 3 and image.shape[2] == 12:  # Multispectral with 12 bands
            image = np.transpose(image, (2, 0, 1))  # Reorder to (channels, height, width)
        elif len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=0)  # Add channel dimension
            image = np.tile(image, (12, 1, 1))  # Convert grayscale to 12 channels
        else:
            raise ValueError("Unexpected image shape.")

        # Normalize the image
        image = normalize_image(image)  # Custom normalization function
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return image
    else:
        raise ValueError("Unsupported file format.")


def normalize_image(image):
    # Normalize each channel separately
    for i in range(image.shape[0]):
        channel = image[i, :, :]
        image[i, :, :] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    return image

# Helper function for post-processing the output
def postprocess_output(output):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()  # Threshold to get binary mask
    output = output.squeeze().cpu().numpy()  # Convert to numpy
    return output

# Helper function to convert image to base64
def image_to_base64(image):
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file:
        # Save the uploaded file temporarily
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        try:
            # Preprocess the image
            input_image = preprocess_image(filepath)

            # Forward pass through the model
            with torch.no_grad():
                output = forward(input_image)

            # Post-process the output
            mask = postprocess_output(output)

            # Convert the mask to an image
            mask_image = (mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_image[0], mode='L')  # Mode 'L' for grayscale

            # Convert original image to RGB using PIL
            original_image_pil = Image.open(filepath).convert("RGB")

            # Convert images to base64
            encoded_original_img = image_to_base64(original_image_pil)
            encoded_mask_img = image_to_base64(mask_pil)

            return jsonify({
                "original_image": encoded_original_img,
                "predicted_mask": encoded_mask_img
            })
        except Exception as e:
            return str(e), 500

    return "Something went wrong", 500

if __name__ == '__main__':
    # Ensure uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True, use_reloader=False)
