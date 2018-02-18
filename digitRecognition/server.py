from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage
import pickle

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # refers to application_top
MODEL_PATH = os.path.join(APP_ROOT, 'model.pkl')


def printImage(image):
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


@app.route('/handleImage', methods=['POST'])
def handleImage():
    if request.method == 'POST':
        # Get the canvas pixels and transform it to array
        canvasPixels = np.array(str(request.data.decode('utf-8')).split(',')).astype(int)

        if all(i == 0 for i in canvasPixels):
            return "Noting because you haven't drawn anything"

        # Divide each pixel value by 10 - that is what the classifier is trained on
        canvasPixels = np.array([i // 10 for i in canvasPixels])

        # Transform to 280x280 image
        canvasImage = canvasPixels.reshape(280, 280)

        # Scale the image to fit the model desired input
        scaledImage = np.array(scipy.ndimage.zoom(canvasImage, 0.1, order=0))

        # Predict
        return str(model.predict(scaledImage.reshape(1, -1))[0])

    else:
        return 'Bad request!'


if __name__ == '__main__':
    model = loadModel(MODEL_PATH)
    app.run(host='0.0.0.0', port=1212, debug=True)
