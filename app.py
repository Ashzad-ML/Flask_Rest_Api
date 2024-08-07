from utils.prediction_function import predict_image, preprocess_image
from flask_restx import Resource, Api, reqparse
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
import secrets
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from database import session, APILog, User

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/predict": {"origins": "http://localhost:3000"},
        r"/base64_image_predict": {"origins": "http://localhost:3000"},
    },
)

api = Api(app, version="1.0", title="Rest API", description="My machine learning API")


def log_api_call(
    endpoint,
    api_key_id,
    request_method,
    status_code,
    prediction_result=None,
    error_message=None,
):
    log_entry = APILog(
        endpoint=endpoint,
        api_key_id=api_key_id,
        request_method=request_method,
        status_code=status_code,
        prediction_result=prediction_result,
        error_message=error_message,
        success=status_code == 200,
    )
    session.add(log_entry)
    session.commit()


def get_user_from_api_key(api_key):
    return session.query(User).filter_by(api_key=api_key).first()


parser = reqparse.RequestParser()
parser.add_argument(
    "image", type=FileStorage, location="files", required=True, help="Image file"
)
parser.add_argument(
    "X-API-Key", type=str, location="headers", required=False, help="API Key"
)


@api.route("/predict")
class Predict(Resource):
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        file = args["image"]
        api_key = args.get("X-API-Key")

        if not api_key:
            error_message = "Missing API key"
            log_api_call(
                request.endpoint, None, request.method, 401, error_message=error_message
            )
            return {"message": error_message}, 401

        user = get_user_from_api_key(api_key)
        if not user:
            error_message = "Invalid API key"
            log_api_call(
                request.endpoint, None, request.method, 401, error_message=error_message
            )
            return {"message": error_message}, 401

        api_key_id = user.id
        status_code = 200
        prediction_result = None
        error_message = None

        if file:
            if file.mimetype not in ["image/jpeg", "image/png", "image/webp"]:
                status_code = 400
                error_message = "Only image files are allowed"
                log_api_call(
                    request.endpoint,
                    api_key_id,
                    request.method,
                    status_code,
                    error_message=error_message,
                )
                return {"message": error_message}, status_code

            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            percentages, result = predict_image(image)
            prediction_result = f"Percentages: {percentages}, Result: {result}"

            log_api_call(
                request.endpoint,
                api_key_id,
                request.method,
                status_code,
                prediction_result,
            )
            return jsonify({"percentages": percentages, "result": result})

        status_code = 400
        error_message = "Image file is empty"
        log_api_call(
            request.endpoint,
            api_key_id,
            request.method,
            status_code,
            error_message=error_message,
        )
        return {"message": error_message}, status_code


baseparser = reqparse.RequestParser()
baseparser.add_argument(
    "image", type=str, location="json", required=True, help="Base64 encoded image"
)
baseparser.add_argument(
    "X-API-Key", type=str, location="headers", required=False, help="API Key"
)


@api.route("/base64_image_predict")
class BasePredict(Resource):
    @api.expect(baseparser)
    def post(self):
        args = baseparser.parse_args()
        image_base64 = args["image"]
        api_key = args.get("X-API-Key")

        if not api_key:
            error_message = "Missing API key"
            log_api_call(
                request.endpoint, None, request.method, 401, error_message=error_message
            )
            return {"message": error_message}, 401

        user = get_user_from_api_key(api_key)
        if not user:
            error_message = "Invalid API key"
            log_api_call(
                request.endpoint, None, request.method, 401, error_message=error_message
            )
            return {"message": error_message}, 401

        api_key_id = user.id
        status_code = 200
        prediction_result = None
        error_message = None

        if image_base64:
            if not (
                image_base64.startswith("data:image/jpeg;base64")
                or image_base64.startswith("iVBORw0KGgo")
                or image_base64.startswith("/9j/")
            ):
                status_code = 400
                error_message = "Only Base64 format files are allowed"
                log_api_call(
                    request.endpoint,
                    api_key_id,
                    request.method,
                    status_code,
                    error_message=error_message,
                )
                return {"message": error_message}, status_code

            if len(request.json) > 1:
                status_code = 400
                error_message = "Extra parameter(s) not allowed"
                log_api_call(
                    request.endpoint,
                    api_key_id,
                    request.method,
                    status_code,
                    error_message=error_message,
                )
                return {"message": error_message}, status_code

            if image_base64.startswith("data:image/jpeg;base64"):
                start = image_base64.index(",") + 1
                image_base64 = image_base64[start:]

            try:
                image_data = base64.b64decode(image_base64, validate=True)
            except (base64.binascii.Error, ValueError):
                status_code = 400
                error_message = "Provided string is not a valid Base64 encoded image"
                log_api_call(
                    request.endpoint,
                    api_key_id,
                    request.method,
                    status_code,
                    error_message=error_message,
                )
                return {"message": error_message}, status_code

            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            percentages, result = predict_image(image)
            prediction_result = f"Percentages: {percentages}, Result: {result}"

            log_api_call(
                request.endpoint,
                api_key_id,
                request.method,
                status_code,
                prediction_result,
            )
            return jsonify({"percentages": percentages, "result": result})

        status_code = 400
        error_message = "Base64 String is empty"
        log_api_call(
            request.endpoint,
            api_key_id,
            request.method,
            status_code,
            error_message=error_message,
        )
        return {"message": error_message}, status_code


checkparser = reqparse.RequestParser()
checkparser.add_argument(
    "X-API-Key", type=str, location="headers", required=True, help="API Key"
)


@api.route("/check_stats")
class CheckStats(Resource):
    @api.expect(checkparser)
    def get(self):
        args = checkparser.parse_args()
        api_key = args.get("X-API-Key")
        user = get_user_from_api_key(api_key)
        if not user:
            return {"message": "Invalid API key"}, 401

        total_successful = (
            session.query(APILog).filter_by(success=True, api_key_id=user.id).count()
        )
        total_unsuccessful = (
            session.query(APILog).filter_by(success=False, api_key_id=user.id).count()
        )
        return jsonify(
            {
                "total_successful": total_successful,
                "total_unsuccessful": total_unsuccessful,
            }
        )


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(debug=True, port=8080)
