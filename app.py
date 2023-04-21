from flask import Flask, request, jsonify, render_template
from fastai.vision.all import *
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load the trained model
path = Path(__file__).parent
model_path = path/'jordan_classifier.pkl'
learn = load_learner(model_path)
print('Input shape:', learn.dls[0].after_item.size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    file = request.files['image']
    img_bytes = file.read()

    # Convert image to PIL.Image format
    try:
        img = Image.open(io.BytesIO(img_bytes))
        print('Image format:', img.format)

    except Exception as e:
        print('Error opening image:', e)
        return jsonify({'error': 'Error opening image'})

    # Debugging: Print the size and mode of the image
    print('Image size:', img.size)
    print('Image mode:', img.mode)

    # Ensure that the image has the correct dimensions
    try:
        img = img.crop_pad((192, 192))
    except Exception as e:
        print('Error cropping/resizing image:', e)
        return jsonify({'error': 'Error cropping/resizing image'})

    # Make a prediction
    try:
        label, index, probs = learn.predict(img)
        confidence = probs[index].item() * 100
    except Exception as e:
        print('Error making prediction:', e)
        return jsonify({'error': 'Error making prediction'})

    # Return the prediction and confidence percentage as a response
    return jsonify({'prediction': str(label), 'confidence': f'{confidence:.2f}%'})




if __name__ == '__main__':
    app.run(debug=True)
