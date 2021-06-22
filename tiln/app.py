import base64
import json
import os
import uuid

import cv2
import flask
from flask import Flask
from flask import request
from werkzeug.utils import secure_filename

from flask import send_file
from colab_interface import test_comment
from censor_image import test_image
from flask_cors import CORS, cross_origin

app = Flask(__name__)
#CORS(app, resources={r"/text-input": {"origins": ["http://localhost:3000"]}})
CORS(app)


@app.route('/hello/', methods=['GET', 'POST'])
@cross_origin()
def welcome():
    return "Hello World!"


@app.route('/text-input', methods=['POST'])
@cross_origin()
def rate_text_input():
    data = json.loads(request.data)
    comment = data.get("comment")
    print(comment)
    model = data.get("model")
    output, impact_words = test_comment(comment, model)
    result = flask.jsonify({
        "comment": comment,
        "output": output,
        "model": model,
        "impact_words": impact_words
    })
    return result


@app.route('/image', methods=['POST'])
@cross_origin()
def rate_image():
    model = request.form.get('model')
    f = request.files['file']
    output, image = test_image(f, model)
    save_father_path = 'imgfiles'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    cv2.imwrite(img_path, image)

    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    base64_string = jpg_as_text.decode('utf-8')
    print(jpg_as_text)
    result = flask.jsonify({
        "file": base64_string
    })
    return result


@app.route('/csv-file', methods=['POST'])
@cross_origin()
def rate_csv_file():
    f = request.files['file']
    save_father_path = 'txtfiles'
    csv_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(csv_path)


    model = request.form.get('model')
    comments = []

    csv_file = open(csv_path, 'r', encoding='utf-8')
    comment_list = csv_file.readlines()
    comment_join = ' '.join(comment_list)
    output_all = test_comment(comment_join, model)
    raw_result = {
        "results": [],
        "output_all": output_all
    }
    print(comment_list)
    for linie in comment_list:
        linie = linie.strip()

        comments.append(linie)
        output, impact_words = test_comment(linie, model)

        raw_result['results'].append({
        "comment": linie,
        "output": output,
        "model": model,
        "impact_words": impact_words,
        })

    result = flask.jsonify(raw_result)

    return result
    
if __name__ == '__main__':
    app.run(host='localhost', port=105)