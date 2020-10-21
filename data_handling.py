
from pattern_matcher.dto.triggering_pattern_dto import TriggeringPatternDTO

# TODO 지식 로드 변경하는 부분 정리되면 변경 필요
def get_triggering_dto_list(project_id):
    # "query": "집에서 판교역가려면 몇번 버스타야돼요?"
    dto_id = 35872920
    dialog_task_id = "35872918"
    order = 1
    pattern_type = "combination"
    pattern = "{{집|회사}},{{서울역|판교역}},{{버스|지하철}}"
    triggering_pattern_dto = TriggeringPatternDTO(dto_id, dialog_task_id, order, pattern_type, pattern)
    triggering_pattern_dto_list = [triggering_pattern_dto]
    return triggering_pattern_dto_list
