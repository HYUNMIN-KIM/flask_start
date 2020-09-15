from flask import Flask, request, make_response
import cnn_module

from flask import json
import threading
app = Flask(__name__)


@app.route('/result', methods=['POST'])
def get_result():
    params = request.get_json()['query']

    result = cnn_module.sentence_classification(params)
    keys = list(result.keys())

    print("key  :", keys)
    id = result[keys[0]]
    return {'intent_id': str(id), 'sentence':keys[0]}



@app.route('/train', methods=['POST'])
def retrain():
    if cnn_module.train_status():
        cnn_module.train()


        return {'train':'success'}
    else:
        return{'train':'fail'}



@app.route('/api/status', methods=['GET'])
def status():
    if cnn_module.train_status():
        return {'status': 'READY'}
    else:
        return {'status': 'BUSY'}

if __name__ == '__main__':
    app.run()
