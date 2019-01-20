import numpy as np


def json_default(o):
    print(o)
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    
    type_name = o.__class__.__name__
    raise TypeError("Object of type {} is not JSON serializable".format(type_name))

