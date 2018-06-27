def map(func, obj):
    if isinstance(obj, dict):
        return {k: func(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [func(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple([func(v) for v in obj])
    raise TypeError("Not supported of map.")
