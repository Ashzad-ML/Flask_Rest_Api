from flask import Flask, request, jsonify
from flask_restx import Resource, Api, reqparse
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import io
import cv2
from PIL import Image
import os
import base64
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage   
from utils.prediction_function import predict_image,preprocess_image


app = Flask(__name__)

# CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
CORS(app, resources={
    r"/predict": {"origins": "http://localhost:3000"},
    r"/base64_image_predict": {"origins": "http://localhost:3000"}
})

api = Api(app, version='1.0', title='Rest Api',
          description='My machine learning API')

# @api.route('/hello')
# class HelloWorld(Resource):
#     def get(self):
#         return {'hello': 'world'}

parser = reqparse.RequestParser()
parser.add_argument('image', type=FileStorage, location='files', required=True, help='Image file')  

@api.route('/predict')
class Predict(Resource):
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        file = args['image']
        
        if file:
            if file.mimetype not in ['image/jpeg', 'image/png', 'image/webp']:
                return {'message': 'Only image files are allowed'}, 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            percentages, result = predict_image(image)

            return jsonify({
                'percentages': percentages,
                'result': result
            })

        return {'message': 'Image file is empty'}, 400

baseparser = reqparse.RequestParser()
baseparser.add_argument('image', type=str, location='json', required=True, help='Base64 encoded image')  

@api.route('/base64_image_predict')
class BasePredict(Resource):
    @api.expect(baseparser)
    def post(self):
        args = baseparser.parse_args()
        image_base64 = args['image']
        
        if image_base64:
            if not image_base64.startswith (( 'data:image/jpeg;base64','iVBORw0KGgo', '/9j/')):
                return {'message': 'Only Base64 format files are allowed'}, 400
            
            if len(request.json) > 1:
                return {'message': 'Extra parameter(s) not allowed'}, 400
        
            if  image_base64.startswith (( 'data:image/jpeg;base64')):
                start = image_base64.index(',') + 1
                image_base64 = image_base64[start:]
                
            # if not base64.b64decode(image_base64, validate=True):
            #      return {'message': 'Provided string is not a valid Base64 encoded image'}, 400
            
            try:
                image_data = base64.b64decode(image_base64, validate=True)
            except (base64.binascii.Error, ValueError):
                return {'message': 'Provided string is not a valid Base64 encoded image'}, 400
                
            image_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            percentages, result = predict_image(image)

            return jsonify({
                'percentages': percentages,
                'result': result
            })

        return {'message': 'Base64 String is empty'}, 400


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    app.run(debug=True, port=8080)