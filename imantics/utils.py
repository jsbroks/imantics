import numpy as np


def json_default(o):
    print(o)
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    
    type_name = o.__class__.__name__
    raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

