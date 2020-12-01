from flask import Flask, request, jsonify
import torch
import os
import flask
from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    data = {}
    return flask.render_template("index.html", data=data)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.files)
        file = request.files.read()

        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        # try:
        # print(file.filename)
        print("reachd")
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        length_digit, output_number = get_prediction(tensor)
        print(length_digit, output_number)
        data = {'prediction': output_number, 'Length': str(length_digit)}
        return flask.render_template("index.html", data=data)

        # except:
        #     return jsonify({'error': 'error during prediction'})

@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = load_image_url(url)
        tensor = transform_image(img)
        length_digit, output_number = get_prediction(tensor)

    else:
        img_bytes = flask.request.files['file'].read()
        tensor = transform_image(img_bytes)
        length_digit, output_number = get_prediction(tensor)
        # print(length_digit, output_number)

    data = {'prediction': output_number, 'Length': str(length_digit)}
    return flask.jsonify(data)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_request():
    app.jinja_env.cache = {}

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.debug = True
    app.run()
