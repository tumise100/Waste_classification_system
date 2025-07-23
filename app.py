import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('waste_classifier_model.h5')

# Define image dimensions and class labels
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Make sure this order is the same as in your training
CLASS_LABELS = ['plastic', 'glass', 'organic', 'paper', 'cardboard', 'metal'] 

def preprocess_image(image_bytes):
    """
    Takes an image in bytes, preprocesses it, and prepares it for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image and returns a prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    try:
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(preprocessed_image)
        
        # Decode prediction
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_label = CLASS_LABELS[predicted_class_index]
        confidence_score = float(np.max(prediction[0]))
        
        return jsonify({
            'prediction': predicted_class_label,
            'confidence': f'{confidence_score:.2f}'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)