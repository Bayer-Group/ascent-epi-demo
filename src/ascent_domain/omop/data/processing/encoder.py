import json
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd


class ExtendedEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()  # Convert to ISO 8601 string format
        elif isinstance(obj, date):
            return obj.isoformat()  # Convert to ISO 8601 string format for date objects
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(ExtendedEncoder, self).default(obj)

