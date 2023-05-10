# backend.py (Python Flask example)

from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from PIL import Image

app = Flask(__name__)
image_caption = pipeline('image-to-text')

# Define the route for file upload
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']
    print(file)

    selected_model = request.form['model']
    img = Image.open(request.files['file'].stream)

    caption = image_caption(img)

    return jsonify({'caption': caption})

@app.route('/', methods=['GET'])
def main():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)