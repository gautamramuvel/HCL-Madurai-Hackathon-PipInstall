# from flask import Flask, render_template, request

from flask import Flask,render_template, request,json
from sklearn.externals import joblib

# import json

app = Flask(__name__)

clf =  joblib.load('model_train1.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def api():
    return "Hello"


if __name__ == '__main__':
    app.run(debug=True)
