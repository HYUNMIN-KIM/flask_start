"""
 Project 전체에 대한 DTO
 사이즈가 매우 큼
 지식관리및학습서버와 대화작업서버간의 데이터 교환을 위해 사용됨
"""

from pattern_matcher.dto import triggering_pattern_dto
class ProjectDTO:
    def __int__(self):
        self.triggering_pattern_dto_list = triggering_pattern_dto()

    # getter
    @property
    def triggering_pattern_dto_list(self):
        return self.triggering_pattern_dto_list

    @triggering_pattern_dto_list.setter
    def triggering_pattern_dto_list(self, triggering_pattern_dto_list):
        self.triggering_pattern_dto_list = triggering_pattern_dto_list
