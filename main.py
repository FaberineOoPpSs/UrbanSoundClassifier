# -*- coding: utf-8 -*-
"""
author: Yashaswi Aryan
reg. no.: 200968186
batch: 4
"""

# Importing libraries
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from flask import Flask, render_template, redirect, request
import os
import classPredictor 


# Creating app object
app = FastAPI()
flask_app = Flask(__name__)

app.mount("/file", WSGIMiddleware(flask_app))

@flask_app.get('/')
def accept_files():
    f = request.files["file"]
    print(f.filename)
    path="./static/" + f.filename
    f.save(path)
    #caption = get_prediction(path)
    caption = classPredictor.predictClass(path)
    os.remove(path)
    return caption

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    #f = request.files["file"]
    #print(f.filename)
    #path = "./static/" + file.filename
    #file.save(path)
    fl = classPredictor.predictClass(path)    
    #return {"filename": file.filename}
    return fl
# Run the api with uvicorn
#if __name__ == '__main__':
#    uvicorn.run(app, host = '127.0.0.1', port = 8000)
    
#uvicorn main:app --reload