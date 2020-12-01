from flask import Flask, request, jsonify
import torch
from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    print('success')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')

        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        # try:
        # print(file.filename)
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        length_digit, output_number = get_prediction(tensor)
        print(length_digit, output_number)
        data = {'prediction': output_number, 'Length': str(length_digit)}

        return jsonify(data)
        # except:
        #     return jsonify({'error': 'error during prediction'})


if __name__ == '__main__':
    app.debug = True
    app.run()
