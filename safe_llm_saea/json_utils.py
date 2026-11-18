import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder that converts NumPy data types to native Python types
    for JSON serialization.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
