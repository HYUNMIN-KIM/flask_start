from enum import Enum


class PatternType(Enum):
    EQUAL = "equal"
    START_WITH = "startWith"
    END_WITH = "endWith"
    COMBINATION = "combination"
    UNDEFINED = "undefined"

    @staticmethod
    def check_type(pattern_type):
        patterns = [e.value for e in PatternType]
        if pattern_type not in patterns:
            return PatternType.UNDEFINED.value

        return pattern_type


