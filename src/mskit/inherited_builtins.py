class NonOverwriteDict(dict):
    def __setitem__(self, key, value):
        if self.__contains__(key):
            pass
        else:
            dict.__setitem__(self, key, value)
