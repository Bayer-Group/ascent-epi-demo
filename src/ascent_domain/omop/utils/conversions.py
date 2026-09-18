"""
Utility functions for data type conversions.
"""

import json
from decimal import Decimal

import numpy as np


def convert_numpy_to_python(obj):
    """
    Convert non-serializable objects to Python native types.

    Handles:
    - NumPy arrays and scalar types
    - NetworkX graphs
    - Other complex objects

    Args:
        obj: Any object that might contain non-serializable types

    Returns:
        Object with all non-serializable types converted to Python native types
    """

    # Handle None
    if obj is None:
        return None

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(i) for i in obj]

    # Handle numpy scalar types
    elif type(obj).__module__ == 'numpy':
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        else:
            # For any other numpy type, convert to string
            return str(obj)

    # Handle NetworkX graphs
    elif type(obj).__module__.startswith('networkx'):
        try:
            # Convert graph to dictionary representation
            if hasattr(obj, 'nodes') and hasattr(obj, 'edges'):
                return {
                    'nodes': list(obj.nodes()),
                    'edges': list(obj.edges()),
                    'type': type(obj).__name__
                }
            else:
                return str(obj)
        except Exception:
            # Fallback if conversion fails
            return str(obj)

    # Handle objects with to_dict method (e.g., UncertaintyResult dataclass)
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return convert_numpy_to_python(obj.to_dict())

    # Handle Enum types
    elif hasattr(obj, 'value') and type(obj).__bases__ and any('Enum' in str(b) for b in type(obj).__mro__):
        return obj.value

    # Return other types unchanged
    else:
        return obj

# Custom JSON encoder to handle Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


# Custom JSON encoder to handle date objects