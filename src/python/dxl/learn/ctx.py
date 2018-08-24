class GlobalContext:
    data = {}

    @classmethod
    def register(cls, key, value):
        cls.data[key] = value

    @classmethod
    def get(cls, key):
        return cls.data.get(key)

    @classmethod
    def reset(cls, key):
        if key in cls.data:
            del cls.data[key]
