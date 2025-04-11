from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
import tensorflow as tf
import os
import pickle  # For loading the label decoder

app = Flask(__name__)

# Load the tokenizer
tokenizer_path = "./model"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Load the saved TensorFlow model
model_directory = os.path.join(tokenizer_path, 'bert_model')
custom_model = tf.saved_model.load(model_directory)

# Load the label decoder
label_decoder_path = os.path.join(tokenizer_path, 'label_encoder.pkl')
with open(label_decoder_path, 'rb') as file:
    label_decoder = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

import logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    logging.debug(f'Input text: {text}')
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='tf', padding='max_length', truncation=True, max_length=128)
    
    logging.debug(f'Tokenized inputs: {inputs}')
    
    # Extract individual inputs
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    logging.debug(f'Input IDs: {input_ids}')
    logging.debug(f'Attention Mask: {attention_mask}')
    
    # Prepare inputs as a list of tensors
    model_inputs = [input_ids, attention_mask]
    
    # Make predictions
    logits = custom_model(model_inputs, training=False)  # Pass inputs as a list
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]  # Extract the scalar value
    
    logging.debug(f'Predicted class: {predicted_class}')
    
    # Decode the label using array indexing
    predicted_label = label_decoder[int(predicted_class)]
    
    logging.debug(f'Predicted label: {predicted_label}')
    
    # Convert prediction to a human-readable format with a gentle message
    response = {
        'text': text,
        'prediction': f"It seems you might be experiencing symptoms related to {predicted_label}. Please remember that seeking support from a professional can be very helpful."
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
