import torch as T
def get_device()->T.device:
    return T.device('cuda:0' if T.cuda.is_available() else 'cpu')