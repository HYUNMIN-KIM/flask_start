from pattern_matcher.pattern.pattern_type import PatternType

class TriggeringPatternDTO:

    def __init__(self, dto_id=None, dialog_task_id=None, order=None, pattern_type=None, pattern=None):
        self._dto_id = dto_id
        self._dialog_task_id = dialog_task_id
        self._order = order
        self._pattern_type = pattern_type
        self._pattern = pattern

    def convert_json_to_object(self, content):
        self._dto_id = content['id']
        self._dialog_task_id = content['dialogTaskId']
        self._order = content['order']
        self._pattern_type = content['type']
        self._pattern = content['pattern']

    @property
    def dto_id(self):
        return self._dto_id

    @dto_id.setter
    def dto_id(self, dto_id):
        self._dto_id = dto_id

    @property
    def dialog_task_id(self):
        return self._dialog_task_id

    @dialog_task_id.setter
    def dialog_task_id(self, dialog_task_id):
        self._dialog_task_id = dialog_task_id

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order

    @property
    def pattern_type(self):
        return PatternType.check_type(self._pattern_type)

    @pattern_type.setter
    def pattern_type(self, pattern_type):
        self._pattern_type = pattern_type

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        self._pattern = pattern

    def isEmpty(self):
        if not self.pattern:
            return True
        else:
            return False
