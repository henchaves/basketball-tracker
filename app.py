from flask import Flask
from flask_restful import Api
from flask_cors import CORS

from resources.upload_video import UploadVideo


def create_app():
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(UploadVideo, "/")
    
    CORS(app)
    return app
