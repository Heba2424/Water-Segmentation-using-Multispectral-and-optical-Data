from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tifffile
import segmentation_models_pytorch as smp

# Initialize Flask app
app = Flask(__name__)

# Load model and pre_conv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'D:\Skills\Traning\Cellula Tec\Stellite Water Segmentation\Deploy\unet_model.pth'

class UNetModel:
    def __init__(self, model_path):
        self.pre_conv = nn.Conv2d(12, 3, kernel_size=1).to(device)
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.pre_conv.load_state_dict(checkpoint['pre_conv_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        x = self.pre_conv(x)
        return self.model(x)

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(device)
            output = self.forward(image)
            return torch.sigmoid(output).cpu().numpy()

# Initialize model
unet_model = UNetModel(model_path)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def transform_image(image):
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def save_plot(inputs_np, output, result):
    buf = io.BytesIO()
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))  # Increase figure size for better visibility
    
    # Handle the inputs_np (original image)
    if inputs_np.shape[2] in [1, 3]:
        axes[0].imshow(inputs_np)
    else:
        # Visualize the first channel if multi-channel
        axes[0].imshow(inputs_np[:, :, 0], cmap='gray')
    axes[0].set_title("Original Image", fontsize=16)  # Increase title font size
    axes[0].axis('off')
    
    # Handle the output (predicted mask)
    if len(output.shape) == 3 and output.shape[2] in [1, 3]:
        # If output is (H, W, C) with C being 1 or 3
        axes[1].imshow(output[:, :, 0])  # Display the first channel
    elif len(output.shape) == 2 or (len(output.shape) == 3 and output.shape[0] == 1):
        # If output is (H, W) or (1, H, W), treat it as grayscale
        axes[1].imshow(np.squeeze(output), cmap='gray')  # Remove single channel dimension
    else:
        axes[1].imshow(output[:, :, 0], cmap='gray')  # Fallback
    axes[1].set_title("Predicted Mask", fontsize=16)  # Increase title font size
    axes[1].axis('off')

    # Handle the result (combined or overlay image)
    if len(inputs_np.shape) == 3 and inputs_np.shape[2] in [1, 3]:
        axes[2].imshow(inputs_np)
        axes[2].imshow(np.squeeze(result), alpha=0.5, cmap='jet')  # Overlay mask
    elif len(inputs_np.shape) == 2 or (len(inputs_np.shape) == 3 and inputs_np.shape[0] == 1):
        axes[2].imshow(inputs_np[:, :, 0], cmap='gray')  # Show first channel
        axes[2].imshow(np.squeeze(result), alpha=0.5, cmap='jet')  # Overlay mask
    else:
        axes[2].imshow(inputs_np[:, :, 0], cmap='gray')  # Fallback
        axes[2].imshow(np.squeeze(result), alpha=0.5, cmap='jet')  # Overlay mask
    axes[2].set_title("Overlay Result", fontsize=16)  # Increase title font size
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join('uploads', filename)
        file.save(img_path)
        
        # Handle TIFF images
        try:
            if filename.lower().endswith('.tif'):
                image = tifffile.imread(img_path)
            else:
                image = Image.open(img_path)
                image = np.array(image)
        except Exception as e:
            return f"Error reading image: {str(e)}"

        # Print image information for debugging
        print("Image type:", type(image))
        print("Image shape:", image.shape)

        # Process multi-channel images
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 12:
                # Convert multi-channel image to a suitable format
                image = image.transpose(2, 0, 1)  # Change to CxHxW
                image = torch.tensor(image).float()  # Convert to tensor
                image = image.unsqueeze(0)  # Add batch dimension
            elif len(image.shape) == 3 and image.shape[2] in [1, 3]:
                # Handle grayscale or RGB images
                image = torch.tensor(image).float()
                image = image.permute(2, 0, 1).unsqueeze(0)  # CxHxW + batch dimension
            else:
                return f"Unsupported image format with shape: {image.shape}"
        else:
            return "Unsupported image format"

        # Ensure image is on the right device
        image = image.to(device)

        # Predict
        output = unet_model.predict(image)
        
        # Post-process
        if isinstance(output, torch.Tensor):
            output = output.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and move to CPU
        elif isinstance(output, np.ndarray):
            output = output.squeeze(0)  # Remove batch dimension if necessary
        else:
            return "Unsupported output format"
        
        # Convert output to a suitable format for saving/displaying
        output = (output > 0.5).astype(np.uint8) * 255  # Convert to binary image
        
        # Save results
        inputs_np = np.squeeze(image.cpu().detach().numpy()).transpose(1, 2, 0)  # CxHxW to HxWxC
        buf = save_plot(inputs_np, output, output)
        buf.seek(0)

        # Save the result image to static folder
        result_path = os.path.join('static', 'result.png')
        with open(result_path, 'wb') as f:
            f.write(buf.getvalue())

        return render_template('result.html', image_url=url_for('static', filename='result.png'))

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
