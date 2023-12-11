from flask import Flask, render_template, make_response
from flask_restful import Resource, reqparse
import werkzeug

from datetime import datetime
import os

from libs.predict_video import predict_video
# from libs.predict_image import load_image, request_model

class UploadVideo(Resource):
    parse = reqparse.RequestParser()
    parse.add_argument("file", type=werkzeug.datastructures.FileStorage, location="files")
    headers = {"Content-Type": "text/html"}

    @classmethod
    def post(cls):
        files = cls.parse.parse_args()
        video = files["file"]

        if "video" in video.content_type:
            time_str = datetime.now().strftime("%Y%m%d%H%M%S")
            video_filename = f"{time_str}-{video.filename}"
            video_path = os.path.join("uploads", video_filename)
            video.save(video_path)
            video_path = predict_video(video_path)
            return make_response(render_template("index.html", message="Upload a video", video_path=video_path), 200, cls.headers)

            return {"message": "Video processed"}, 200
        
        return {"message": "Error"}, 400
    
    @classmethod
    def get(cls):
        return make_response(render_template("index.html", message="Upload a video", video_path=False), 200, cls.headers)
