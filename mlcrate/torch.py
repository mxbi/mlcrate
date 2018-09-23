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
    NOTE: If the input is a torch tensor, the type will not be checked.

    Keyword arguments:
    arr -- Any array-like object (eg numpy array, list, numpy varaible)
    device (optional) -- Move the tensor to this device after creation
    type -- the numpy data type of the tensor. Defaults to 'float32' (regardless of the input)

    Returns:
    tensor - A torch tensor"""
    _check_torch_import()
    # If we're given a tensor, send it right back.
    if isinstance(arr, torch.Tensor):
        if device:
            return arr.to(device) # If tensor is already on the specified device, this doesn't copy the tensor.
        else:
            return arr

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

# We define the class only at call-time to allow for lazy torch import
# TODO: Find a less hacky way of doing this. If you have any ideas let me know :)
def MLP(dim, dropout=0.3, hidden_activation='relu', output_activation=None):
    """A fully connected MLP network with the specified sizes at each layer.

    Arguments:
    shape -- A list of the shape of the network. The first item is the input size and the last item is the output size.
             Eg. [128, 512, 512, 1] defines a network with 128 inputs, 1 output and two hidden layers of size 512.
    dropout (default: 0.3) -- Proportion for dropout inbetween layers
    hidden_activation (default: torch.nn.ReLU) -- Activation function between layers.
    output_activation (optional) -- Activation function at the end of the network.
    """
    _check_torch_import()
    
    if hidden_activation == 'relu':
        hidden_activation = torch.nn.ReLU

    class mlc_MLP(torch.nn.Module):
        def __init__(self, dim, dropout, hidden_activation, output_activation):
            super(self, mlc_MLP).__init__()

            self.layers = []
            for f_in, f_out in zip(dim[:-2], dim[1:-1]):
                self.layers += [
                    torch.nn.Linear(f_in, f_out),
                    hidden_activation(),
                    torch.nn.Dropout(dropout),
                ]

            self.layers += [torch.nn.Linear(dim[-2], dim[-1])]

            if output_activation:
                self.layers += [output_activation()]

            self.model = torch.nn.Sequential(*self.layers)

        def forward(self, x):
            return self.model(x)

    return mlc_MLP(dim, dropout, hidden_activation, output_activation)
