from flask import Flask, request, jsonify
from flask_restx import Resource, Api, reqparse
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from werkzeug.datastructures import FileStorage
from utils.prediction_function import predict_image , preprocess_image


app = Flask(__name__)

api = Api(app, version='1.0', title='My Base64 Rest Api',
          description='My machine learning API')

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


baseparser = reqparse.RequestParser()
baseparser.add_argument('image', type=str, location='form', required=True, help='Base64 encoded image')  

@api.route('/base_form_predict')
class BasePredict(Resource):
    @api.expect(baseparser)
    def post(self):
        args = baseparser.parse_args()
        image_base64 = args['image']
        
        if image_base64:
            image_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            percentages, result = predict_image(image)

            return jsonify({
                'percentages': percentages,
                'result': result
            })

        return {'message': 'Image file is empty'}, 400

if __name__ == '__main__':
    app.run(debug=True)