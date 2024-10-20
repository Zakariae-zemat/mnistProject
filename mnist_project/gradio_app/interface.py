import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Load the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Load model weights
model_path = Path(__file__).resolve().parent.parent / "model" / "mnist_ann-model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Function for Gradio to make predictions
def predict_image(img):
    try:
        # Ensure img is a PIL image
        if isinstance(img, Image.Image):
            img = img  # It's already a PIL image
        else:
            img = Image.fromarray(img)  # Convert from numpy array to PIL Image
        
        # Transform the image
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
        
        return f"Predicted Digit: {pred}"
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Gradio interface
iface = gr.Interface(fn=predict_image, inputs="image", outputs="text", title="MNIST Digit Classifier")
iface.launch()
