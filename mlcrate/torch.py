import numpy as np

# We lazy-load torch the first time one of the functions that requires it is called
torch = None

def _check_torch_import():
    global torch
    if torch is not None:
        return
    import importlib
    torch = importlib.import_module('torch')

def tonp(tensor):
    """Takes any PyTorch tensor and converts it to a numpy array or scalar as appropiate.
    Not heavily optimized."""
    arr = tensor.data.detach().cpu().numpy()
    if arr.shape == ():
        return np.asscalar(arr)
    else:
        return arr

def totensor(arr, device=None, type='float32'):
    """Converts any array-like or scalar to a PyTorch tensor, and checks that the array is in the correct type (defaults to float32) and on the correct device.
    Equivalent to calling `torch.from_array(np.array(arr, dtype=type)).to(device)` but more efficient.

    Keyword arguments:
    arr -- Any array-like object (eg numpy array, list, numpy varaible)
    features -- a list of feature names corresponding to the features the model was trained on.

    Returns:
    importance -- A list of (feature, importance) tuples representing sorted importance"""
    _check_torch_import()
    # Only call np.array() if it is not already an array, otherwise numpy will waste time copying the array
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # Likewise with type conversion
    if arr.dtype != type:
        arr = arr.astype(type, copy=False)

    tensor = torch.from_numpy(arr)
    if device:
        tensor = tensor.to(device)
    return tensor
