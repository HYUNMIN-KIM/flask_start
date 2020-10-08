class similarity_data:

    def __init__(self,query):
        self.__query = query

    def __init__(self, id, sentence, score):
        self.id = id
        self.sentence = sentence
        self.score = score



    def info(self):
        super().info()





class analyzed_query:
    def __init__(self,query):
        self.__query = query

    def __init__(self, project_id=None, id=None, query=None, way_of_recommend=None, threshold=None, sim_query_list=None):
        self.__project_id = project_id
        self.__query = query
        self.__id = id
        self.__way_of_recommend = way_of_recommend
        self.__threshold = threshold
        self.__sim_query_list = sim_query_list

    def info(self):
        super().info()

    @property
    def project_id(self):
        return self.__project_id

    @project_id.setter
    def project_id(self, project_id):
        self.__project_id = project_id

    @property
    def query(self):
        return self.__query

    @query.setter
    def query(self, query):
        self.__query = query

    @property
    def way_of_recommend(self):
        return self.__way_of_recommend

    @query.setter
    def way_of_recommend(self, way_of_recommend):
        self.__way_of_recommend =way_of_recommend

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @property
    def sim_query_list(self):
        return self.__sim_query_list

    @sim_query_list.setter
    def sim_query_list(self, sim_query_list):
        self.__sim_query_list= sim_query_list

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id= id
