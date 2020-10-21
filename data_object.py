class similarity_data:

    def __init__(self,query):
        self._query = query

    def __init__(self, id, sentence, score):
        self.id = id
        self.sentence = sentence
        self.score = score



    def info(self):
        super().info()





class analyzed_query:
    def __init__(self,query):
        self.__query = query

    def __init__(self, project_id=None, id=None, query=None,
                       way_of_recommend=None, threshold=None, sim_query_list=None,
                       matched=None, clf_label_list=None, matched_pattern=None):
        self._project_id = project_id
        self._query = query
        self._id = id
        self._way_of_recommend = way_of_recommend
        self._threshold = threshold
        self._sim_query_list = sim_query_list
        self._matched = matched
        self._clf_label_list = clf_label_list
        self._matched_pattern = matched_pattern

    def info(self):
        super().info()

    @property
    def project_id(self):
        return self._project_id

    @project_id.setter
    def project_id(self, project_id):
        self._project_id = project_id

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, query):
        self._query = query

    @property
    def way_of_recommend(self):
        return self._way_of_recommend

    @query.setter
    def way_of_recommend(self, way_of_recommend):
        self._way_of_recommend =way_of_recommend

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def sim_query_list(self):
        return self._sim_query_list

    @sim_query_list.setter
    def sim_query_list(self, sim_query_list):
        self._sim_query_list = sim_query_list

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id= id

    @property
    def matched(self):
        return self._matched

    @matched.setter
    def matched(self, matched):
        self._matched = matched

    @property
    def clf_label_list(self):
        return self._clf_label_list

    @clf_label_list.setter
    def clf_label_list(self, clf_label_list):
        self._clf_label_list = clf_label_list

    @property
    def matcehd_pattern(self):
        return self._matched_pattern

    @matcehd_pattern.setter
    def matched_pattern(self, matched_pattern):
        self._matched_pattern = matched_pattern
