from flask import Flask, request, make_response
import cnn_module
from flask import jsonify
import data_object
import json

from pattern_matcher.dto.triggering_pattern_dto import TriggeringPatternDTO
from pattern_matcher.sentence_pattern_matcher import SentencePatternMatcher

app = Flask(__name__)


@app.route('/intent', methods=['POST'])
def get_result():
    # json example
    # TODO ID 통일 필요
    # {
    #     "project_id": 1,
    #     "threshold": 0.5,
    #     "confident_threshold": 0.2,
    #     "confident_threshold_gap": 0.05,
    #     "id": 35872920,
    #     "dialogTaskId": "35872918",
    #     "order": 1,
    #     "type": "combination",
    #     "pattern": "{{집|회사}},{{서울역|판교역}},{{버스|지하철}}",
    #     "query": "집에서 판교역가려면 몇번 버스타야돼요?"
    # }

    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    params_con_threshold = request.get_json()['confident_threshold']
    params_con_threshold_gap = request.get_json()['confident_threshold_gap']
    aq = data_object.analyzed_query(project_id=1, query=param_query, threshold=params_threshold)

    # TODO 지식 DB에서 데이터 읽어오는 부분 필요
    # 패턴 매칭
    param_triggering_dto = request.get_json()
    triggering_pattern_dto = TriggeringPatternDTO()
    triggering_pattern_dto.convert_json_to_object(param_triggering_dto)
    triggering_pattern_dto_list = [triggering_pattern_dto]
    sentence_pattern = SentencePatternMatcher(triggering_pattern_dto_list)

    sentence_pattern_dto = sentence_pattern.match_sentence(param_triggering_dto['query'])
    aq.way_of_recommend = "PATTERN"
    return json.dumps(sentence_pattern_dto.__dict__, ensure_ascii=False)


    # 유사질의 분석
    sim_list = cnn_module.similarity_top_k(param_query, params_threshold, 5)
    # sim_list_low = cnn_module.similarity_top_k(param_query, 0.5, 5)
    # print("thresold : ",params_threshold)
    # if not sim_list_low:
    #     aq.id = -1
    #     aq.way_of_recommend = "Meaningless"
    #     json_data = json.dumps(aq.__dict__, ensure_ascii=False)
    #     return json_data
    # tmp = []

    if sim_list:
        if params_threshold < sim_list[0].score:
            aq.id = str(sim_list[0].id)
            aq.way_of_recommend = "SIMSENTENCE"
            json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            return json_data
        # for sim in sim_list:
        #     tmp.append({'intent_id': str(sim.id), 'sentence': sim.sentence, 'score': sim.score})
        #
        #     aq.sim_query_list = tmp
        #     aq.way_of_recommend = "SIMSENTENCE"


    print("--------------------sim end------------------")
    clf_result = cnn_module.sentence_classification(param_query)
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
    # TODO ID 통일 필요
    # {
    #     "project_id": 1,
    #     "threshold": 0.5,
    #     "confident_threshold": 0.2,
    #     "confident_threshold_gap": 0.05,
    #     "id": 35872920,
    #     "dialogTaskId": "35872918",
    #     "order": 1,
    #     "type": "combination",
    #     "pattern": "{{집|회사}},{{서울역|판교역}},{{버스|지하철}}",
    #     "query": "집에서 판교역가려면 몇번 버스타야돼요?"
    # }
    param_project_id = request.get_json()['project_id']
    param_query = request.get_json()['query']
    params_threshold = request.get_json()['threshold']
    params_con_threshold = request.get_json()['confident_threshold']
    params_con_threshold_gap = request.get_json()['confident_threshold_gap']
    aq = data_object.analyzed_query(project_id=param_project_id, query=param_query, threshold=params_threshold)

    # TODO 지식 DB에서 데이터 읽어오는 부분 필요
    # 패턴 매칭
    param_triggering_dto = request.get_json()
    triggering_pattern_dto = TriggeringPatternDTO()
    triggering_pattern_dto.convert_json_to_object(param_triggering_dto)
    triggering_pattern_dto_list = [triggering_pattern_dto]
    sentence_pattern = SentencePatternMatcher(triggering_pattern_dto_list)

    sentence_pattern_dto = sentence_pattern.match_sentence(param_triggering_dto['query'])
    aq.way_of_recommend = "PATTERN"
    return json.dumps(sentence_pattern_dto.__dict__, ensure_ascii=False)

    # 유사질의 분석
    result = []

    sim_list = cnn_module.similarity_top_k(param_query, 0, 5)


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

            return json.dumps(result,ensure_ascii=False)
        else:
            aq.matched = False
            json_data = json.dumps(aq.__dict__, ensure_ascii=False)
            result.append(aq.__dict__)
    print(aq.sim_query_list)
    aq = data_object.analyzed_query(project_id=param_project_id, query=param_query, threshold=params_threshold)
    clf_result = cnn_module.sentence_classification(param_query)
    tmp_clf = []
    for index,value in clf_result:
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
