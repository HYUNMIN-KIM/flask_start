from flask import Flask, request, make_response
import intent_recommender
from flask import jsonify
import data_object
import json
import requests
from pattern_matcher.sentence_pattern_matcher import SentencePatternMatcher
import data_handling

app = Flask(__name__)

response = requests.get('http://127.0.0.1:17405/knowledge/project-ids')
print(response.json())
pj_list = response.json()
print(pj_list[0])
intent_recommender.load_simulation(pj_list)
print(intent_recommender.train_status(pj_list[0]))
intent_recommender.load_service(pj_list)


@app.route('/recommend-intent', methods=['POST'])
def get_result():
    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    params_con_threshold = request.get_json()['confident_threshold']
    params_con_threshold_gap = request.get_json()['confident_threshold_gap']
    params_dialogtask_mode = request.get_json()['dialog_task_mode']
    aq = data_object.analyzed_query(project_id=1, query=param_query, threshold=params_threshold)

    # 패턴 매칭
    sentence_pattern_matcher = SentencePatternMatcher(data_handling.get_triggering_dto_list(project_id=1))
    sentence_pattern = sentence_pattern_matcher.match_sentence(param_query)
    if sentence_pattern is not None:
        aq.id = sentence_pattern.label
        aq.way_of_recommend = "PATTERN"
        aq.matched_pattern = sentence_pattern.__dict__
        json_data = json.dumps(aq.__dict__, ensure_ascii=False)
        return json_data

    # 유사질의 분석
    sim_list = intent_recommender.similarity_top_k(param_query, params_threshold, 5, param_project_id)

    if sim_list:
        if params_threshold < sim_list[0].score:
            aq.id = str(sim_list[0].id)
            aq.way_of_recommend = "SIMSENTENCE"
            json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            return json_data

    clf_result = intent_recommender.sentence_classification(param_query, param_project_id)
    if clf_result[0][1] < params_con_threshold and (clf_result[0][1] - clf_result[1][1]) < params_con_threshold_gap:
        aq.id = 0;
        aq.way_of_recommend = "CLASSIFIER"
    else:
        aq.id = str(clf_result[0][0])
        aq.way_of_recommend = "CLASSIFIER"

    json_data = json.dumps(aq.__dict__, ensure_ascii=False)
    print(json_data)
    return json_data


@app.route('/sentence/analyze', methods=['POST'])
def analyze_query():
    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    params_con_threshold = request.get_json()['confident_threshold']
    params_con_threshold_gap = request.get_json()['confident_threshold_gap']
    aq = data_object.analyzed_query(project_id=param_project_id, query=param_query, threshold=params_threshold)

    # 패턴 매칭
    sentence_pattern_matcher = SentencePatternMatcher(data_handling.get_triggering_dto_list(project_id=1))
    sentence_pattern = sentence_pattern_matcher.match_sentence(param_query)
    if sentence_pattern is not None:
        aq.id = sentence_pattern.label
        aq.way_of_recommend = "PATTERN"
        aq.matched_pattern = sentence_pattern.__dict__
        json_data = json.dumps(aq.__dict__, ensure_ascii=False)
        return json_data

    # 유사질의 분석
    result = []

    sim_list = intent_recommender.similarity_top_k(param_query, 0, 5, param_project_id)

    if sim_list:
        tmp = []
        aq.id = str(sim_list[0].id)
        for sim in sim_list:
            tmp.append({'intent_id': str(sim.id), 'sentence': sim.sentence, 'score': sim.score})
        aq.sim_query_list = tmp
        aq.way_of_recommend = "SIMSENTENCE"
        if sim_list[0].score >= params_threshold:

            aq.matched = True
            # json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            print(aq.sim_query_list)
            result.append(aq.__dict__)

            return json.dumps(result, ensure_ascii=False)
        else:
            aq.matched = False
            json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            result.append(aq.__dict__)
    print(aq.sim_query_list)
    aq = data_object.analyzed_query(project_id=param_project_id, query=param_query, threshold=params_threshold)
    clf_result = intent_recommender.sentence_classification(param_query, param_project_id)
    tmp_clf = []
    for index, value in clf_result:
        tmp_clf.append({'id': index, 'method': 'CNN', 'score': value})

    if clf_result[0][1] < params_con_threshold and (clf_result[0][1] - clf_result[1][1]) < params_con_threshold_gap:
        aq.id = 0;
        aq.way_of_recommend = "CLASSIFIER"
        aq.matched = False
    else:
        aq.id = str(clf_result[0][0])
        aq.way_of_recommend = "CLASSIFIER"
        aq.matched = True

    aq.clf_label_list = tmp_clf

    # json_data = json.dumps(aq.__dict__, ensure_ascii=False)
    result.append(aq.__dict__)

    return json.dumps(result, ensure_ascii=False)


@app.route('/sentences', methods=['POST'])
def sim_sentence():
    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    params_top_k = request.get_json()['top_k']
    sim_list = intent_recommender.similarity_top_k(param_query, params_threshold, params_top_k, param_project_id)
    result = []
    for sim in sim_list:
        result.append({'intent_id': str(sim.id), 'sentence': sim.sentence, 'score': sim.score})

    return jsonify(result);


@app.route('/train/start', methods=['GET'])
def retrain():
    param_project_id = request.args.get('projectId')
    param_project_id = int(param_project_id)
    print(param_project_id)
    if intent_recommender.train_status(param_project_id):
        intent_recommender.train(param_project_id)
        return {'train': 'success'}
    else:
        return {'train': 'fail'}


@app.route('/train/status', methods=['GET'])
def status():
    param_project_id = request.args.get('projectId')
    param_project_id = int(param_project_id)
    if intent_recommender.train_status(param_project_id):
        return 'READY'
    else:
        return 'BUSY'


@app.route('/copy/model', methods=['GET'])
def apply():
    param_project_id = request.args.get('projectId')
    param_project_id = int(param_project_id)
    intent_recommender.apply_model(param_project_id)

    return 'SUCCESS'

@app.route('/project', methods=['DELETE'])
def delete_project():
    param_project_id = request.args.get('projectId')
    param_project_id = int(param_project_id)
    if intent_recommender.delete_project(param_project_id):
        return "SUCCESS"
    else:
        return "FAILURE"



if __name__ == '__main__':
    app.run()
