class SentencePattern:
    def __init__(self, label, order, pattern_type, text):
        self._label = label
        self._order = order
        self._type = pattern_type
        self._text = text


    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        self._type = type

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
