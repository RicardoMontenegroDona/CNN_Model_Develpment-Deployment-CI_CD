from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import os

# inicializa Flask
app = Flask(__name__)

# importa e carrega o modelo
from CNN_Architecture import CNN  # substitua pelo seu arquivo com a classe CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = CNN()
model_path = os.path.join(os.getcwd(), "model", "CNN_Model.pth")
cnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
cnn.eval()

# transformações (mesmas do treino)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 tem 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route('/')
def home():
    return render_template('index.html')

# rota para prever
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = transform(img).unsqueeze(0)  # adiciona batch dimension

    with torch.no_grad():
        outputs = cnn(img)
        _, predicted = torch.max(outputs, 1)

    # classes CIFAR10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    return jsonify({'class': classes[predicted.item()]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
