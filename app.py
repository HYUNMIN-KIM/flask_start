from flask import Flask, request, make_response
import cnn_module
from flask import jsonify
import data_object
import json

app = Flask(__name__)


@app.route('/intent', methods=['POST'])
def get_result():
    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    # 유사질의 분석
    aq = data_object.analyzed_query(project_id=1, query=param_query, threshold=params_threshold)
    sim_list = cnn_module.similarity_top_k(param_query, params_threshold, 5)
    sim_list_low = cnn_module.similarity_top_k(param_query, 0.5, 5)

    if not sim_list_low:
        aq.id = -1
        aq.way_of_recommend = "Meaningless"
        json_data = json.dumps(aq.__dict__, ensure_ascii=False)
        return json_data

    if sim_list:
        tmp = []
        aq.id = str(sim_list[0].id)
        for sim in sim_list:
            tmp.append({'intent_id': str(sim.id), 'sentence': sim.sentence, 'score': sim.score})
        aq.sim_query_list = tmp
        aq.way_of_recommend = "SIMSENTENCE"
    else:
        clf_result = cnn_module.sentence_classification(param_query)
        keys = list(clf_result.keys())
        aq.id = str(clf_result[keys[0]])
        aq.way_of_recommend = "CLASSIFY"

    json_data = json.dumps(aq.__dict__, ensure_ascii=False)
    return json_data


@app.route('/sentence/analyze', methods=['POST'])
def analyze_query():
    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    # 유사질의 분석
    result = []

    sim_list = cnn_module.similarity_top_k(param_query, 0, 5)


    if sim_list:
        aq = data_object.analyzed_query(project_id=param_project_id, query=param_query, threshold=params_threshold)
        tmp = []
        aq.id = str(sim_list[0].id)
        for sim in sim_list:
            tmp.append({'intent_id': str(sim.id), 'sentence': sim.sentence, 'score': sim.score})
        aq.sim_query_list = tmp
        if sim_list[0].score >= params_threshold:
            aq.way_of_recommend = "SIMSENTENCE"
            aq.matched = True
            # json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            print(aq.sim_query_list)
            result.append(aq.__dict__)

            return json.dumps(result,ensure_ascii=False)
        else:
            aq.matched = False
            json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            result.append(aq.__dict__)

    aq = data_object.analyzed_query(project_id=param_project_id, query=param_query, threshold=params_threshold)
    clf_result = cnn_module.sentence_classification(param_query)
    keys = list(clf_result.keys())
    aq.id = str(clf_result[keys[0]])
    tmp_clf =[]
    tmp_clf.append({'id': aq.id, 'method': 'CNN', 'score': 1})
    aq.clf_label_list = tmp_clf
    aq.way_of_recommend = "CLASSIFIER"
    aq.matched = True
    # json_data = json.dumps(aq.__dict__, ensure_ascii=False)
    result.append(aq.__dict__)

    return json.dumps(result,ensure_ascii=False)


@app.route('/train', methods=['POST'])
def retrain():
    if cnn_module.train_status():
        cnn_module.train()
        return {'train': 'success'}
    else:
        return {'train': 'fail'}


@app.route('/api/sentences/', methods=['POST'])
def sim_sentence():
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    params_top_k = request.get_json()['top_k']
    sim_list = cnn_module.similarity_top_k(param_query, params_threshold, params_top_k)
    result = []
    for sim in sim_list:
        result.append({'intent_id': str(sim.id), 'sentence': sim.sentence, 'score': sim.score})

    return jsonify(result);


@app.route('/api/status', methods=['GET'])
def status():
    if cnn_module.train_status():
        return {'status': 'READY'}
    else:
        return {'status': 'BUSY'}


if __name__ == '__main__':
    app.run()
