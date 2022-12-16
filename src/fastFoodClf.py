import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = [
    'Baked Potato',
    'Burger',
    'Crispy Chicken',
    'Donut',
    'Fries',
    'Hot Dog',
    'Pizza',
    'Sandwich',
    'Taco',
    'Taquito'
 ]

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# model = model.to(device)
# load model from ./models
model.load_state_dict(torch.load("models/fastFoodClf_ft.pth"))
model.eval()

def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    return confidences

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=10)    
)

# display the app
app.launch()