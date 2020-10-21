from pattern_matcher.bndm import BNDM
from pattern_matcher.dto.triggering_pattern_dto import TriggeringPatternDTO
from pattern_matcher.pattern.sentence_pattern import SentencePattern
from pattern_matcher.pattern.pattern_type import PatternType
import re


class SentencePatternMatcher:
    def __init__(self, triggering_pattern_dto_list):

        self._sentence_pattern_list = []
        self._sentence_pattern_list
        self._triggering_pattern_dto_list = triggering_pattern_dto_list
        self._string_search = BNDM()
        self._pattern = "\\{\\{.*?\\}\\}"
        # print(self._stringSearch.searchString("집에서 판교역가려면 몇번 버스타야돼요?",
        #                                         "판교역"))
        self.load()

    def load(self):
        for labeled_pattern_dto in self._triggering_pattern_dto_list:
            dialog_task_id = labeled_pattern_dto.dialog_task_id
            order = labeled_pattern_dto.order
            pattern_type = labeled_pattern_dto.pattern_type
            pattern = labeled_pattern_dto.pattern
            self._sentence_pattern_list.append(SentencePattern(
                dialog_task_id, order, pattern_type, pattern))

        self._sentence_pattern_list.sort()

    def parse_sentence_to_list(self, sentence):
        ret_list = []
        matcher_iter = re.finditer(self._pattern, sentence)

        cur_token_end_idx = 0
        prev_token_end_idx = 0

        b_matched = False
        for matcher in matcher_iter:
            if matcher.group():
                b_matched = True
                matched_str = matcher.group()
                cur_token_start_idx = matcher.start()
                cur_token_end_idx = matcher.end()

                if cur_token_start_idx > prev_token_end_idx:
                    sub_str = sentence[prev_token_end_idx:cur_token_start_idx]
                    single_str_array = []
                    single_str_array.insert(0, sub_str)
                    ret_list.append(single_str_array)

                filtered_matched_str = matched_str.replace("{{", "").replace("}}", "")
                array = filtered_matched_str.split("|")

                # TODO 동작 체크 필요(현재 동작을 100% 이해하지 못했음)
                if "|}}" in matched_str or "||" in matched_str or "{{|" in matched_str:
                    array_contain_empty_str = ["" for i in range(len(array) + 1)]
                    array_contain_empty_str[0:len(array)] = array[0: len(array)]
                    array_contain_empty_str[len(array)] = ""
                    ret_list.append(array_contain_empty_str)
                else:
                    ret_list.append(array)

                prev_token_end_idx = cur_token_end_idx

        single_str_array = []
        if cur_token_end_idx < (len(sentence) - 1):
            single_str_array.append(sentence[cur_token_end_idx:])
            ret_list.append(single_str_array)
        elif not b_matched:
            single_str_array.append(sentence)
            ret_list.append(single_str_array)

        return ret_list

    def forward_match(self, sentence, prefix_list, str_array):
        ret_list = []
        if prefix_list is None:
            for str_unit in str_array:
                if sentence.startswith(str_unit):
                    ret_list.append(str_unit)
        else:
            for prefix in prefix_list:
                for str_unit in str_array:
                    merged_str = prefix + str_unit
                    if sentence.startswith(merged_str):
                        ret_list.append(merged_str)

        return ret_list

    def backward_match(self, sentence, postfix_list, str_array):
        ret_list = []
        if postfix_list is None:
            for str_unit in str_array:
                if sentence.endswith(str_unit):
                    ret_list.append(str_unit)
        else:
            for postfix in postfix_list:
                for str_unit in str_array:
                    merged_str = str_unit + postfix
                    if sentence.endswith(merged_str):
                        ret_list.append(merged_str)
        return ret_list

    def equal_match(self, sentence, stored_pattern):
        parse_str_list = self.parse_sentence_to_list(stored_pattern)
        matched_strings = None
        for i in range(len(parse_str_list)):
            next_array = parse_str_list[i]

            if i == 0:
                matched_strings = self.forward_match(sentence, None, next_array)
            else:
                matched_strings = self.forward_match(sentence, matched_strings, next_array)

            for matched_string in matched_strings:
                if sentence == matched_string:
                    return True

        return False

    def start_with_match(self, sentence, stored_pattern):
        parse_str_list = self.parse_sentence_to_list(stored_pattern)
        matched_strings = None
        for i in range(len(parse_str_list)):
            next_array = parse_str_list[i]
            if len(next_array) == 1:
                if len(next_array[0]) == 0:
                    continue

            if i == 0:
                matched_strings = self.forward_match(sentence, None, next_array)
            else:
                matched_strings = self.forward_match(sentence, matched_strings, next_array)

        if matched_strings is None:
            return False

        return len(matched_strings) > 0

    def end_with_match(self, sentence, stored_pattern):
        parse_str_list = self.parse_sentence_to_list(stored_pattern)
        matched_strings = None

        for i in range(len(parse_str_list) - 1, -1, -1):
            next_array = parse_str_list[i]

            if i == (len(parse_str_list) - 1):
                matched_strings = self.backward_match(sentence, None, next_array)
            else:
                matched_strings = self.backward_match(sentence, matched_strings, next_array)

        if matched_strings is None:
            return False

        return len(matched_strings) > 0

    def get_word_set_list_form_pattern_str(self, pattern_str):
        token_or_token_group_array = pattern_str.strip().split(",")
        word_set_list = []

        for token_or_token_group in token_or_token_group_array:
            if not token_or_token_group:
                continue

            if token_or_token_group.startswith("{{") and token_or_token_group.endswith("}}"):
                filtered_str = token_or_token_group.replace("}}", "").replace("{{", "")
                token_group = filtered_str.split("|")
                group_set = set()
                for token in token_group:
                    group_set.add(token)

                word_set_list.append(group_set)
            else:
                group_set = set()
                group_set.add(token_or_token_group)
                word_set_list.add(group_set)

        return word_set_list

    def combination_match(self, sentence, stored_pattern):
        word_set_list = self.get_word_set_list_form_pattern_str(stored_pattern)

        if len(word_set_list) < 3:
            return False

        match_count = 0
        for word_set in word_set_list:
            for word in word_set:
                if len(self._string_search.searchString(sentence.lower(), word.lower())) > 0:
                    match_count += 1
                    break

        if match_count >= len(word_set_list):
            return True
        else:
            return False

    def get_sentence_pattern_matcher(self, project_id = None):
        # "query": "집에서 판교역가려면 몇번 버스타야돼요?"
        dto_id = 35872920
        dialog_task_id = "35872918"
        order = 1
        pattern_type = "combination"
        pattern = "{{집|회사}},{{서울역|판교역}},{{버스|지하철}}"
        triggering_pattern_dto = TriggeringPatternDTO(dto_id, dialog_task_id, order, pattern, pattern)
        triggering_pattern_dto_list = [triggering_pattern_dto]
        sentence_pattern = self(triggering_pattern_dto_list)
        return sentence_pattern

    def match_sentence(self, sentence):
        sentence = sentence.strip()
        for sentence_pattern in self._sentence_pattern_list:
            stored_pattern = sentence_pattern.text.strip()

            if sentence_pattern.type == PatternType.EQUAL.value:
                if self.equal_match(sentence, stored_pattern):
                    return SentencePattern(sentence_pattern.label,
                                           sentence_pattern.order,
                                           sentence_pattern.type,
                                           sentence_pattern.text)

            elif sentence_pattern.type == PatternType.START_WITH.value:
                if self.start_with_match(sentence, stored_pattern):
                    return SentencePattern(sentence_pattern.label,
                                           sentence_pattern.order,
                                           sentence_pattern.type,
                                           sentence_pattern.text)

            elif sentence_pattern.type == PatternType.END_WITH.value:
                if self.end_with_match(sentence, stored_pattern):
                    return SentencePattern(sentence_pattern.label,
                                           sentence_pattern.order,
                                           sentence_pattern.type,
                                           sentence_pattern.text)

            elif sentence_pattern.type == PatternType.COMBINATION.value:
                if self.combination_match(sentence, stored_pattern):
                    return SentencePattern(sentence_pattern.label,
                                           sentence_pattern.order,
                                           sentence_pattern.type,
                                           sentence_pattern.text)


if __name__ == '_main_':
    dummy_triggering_pattern_dto_list = []

    # equal
    # dummy_triggering_pattern_dto = TriggeringPatternDTO(35872920, '35872918',
    #                                                     1, 'equal',
    #                                                     '{{집|회사}} 가는 길')

    # star with
    # dummy_triggering_pattern_dto = TriggeringPatternDTO(35872920, '35872918',
    #                                                     1, 'startWith',
    #                                                     '{{집|회사}} 가')

    # end with
    # dummy_triggering_pattern_dto = TriggeringPatternDTO(35872920, '35872918',
    #                                                     1, 'endWith',
    #                                                     '로 {{가는 길|가는 중}}')

    # combination
    dummy_triggering_pattern_dto = TriggeringPatternDTO(35872920, '35872918',
                                                        1, 'combination',
                                                        '{{집|회사}},{{서울역|판교역}},{{버스|지하철}}')

    dummy_triggering_pattern_dto_list.append(dummy_triggering_pattern_dto)
    spm = SentencePatternMatcher(dummy_triggering_pattern_dto_list)

    equal_obj = spm.match_sentence("집 가는 길")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("회사 가는 길")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("회사 가기 싫어")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("집 회사 가는 길")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("집 회사 가는 길")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("집으로 가는 길")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("회사로 가는 중")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("집에서 판교역가려면 몇번 버스타야돼요?")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("회사에서 서울역까지 무슨 지하철타고 가야돼?")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("어떤 버스타야 집에서 판교역까지 가나요?")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("회사에서 서울역까지 어떻게 가?")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

    equal_obj = spm.match_sentence("판교역가려면 몇번 버스 타야돼?")
    if equal_obj is not None:
        print("정답 : " + equal_obj.label)
    else:
        print("매칭 실패")

