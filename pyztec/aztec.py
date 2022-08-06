import numpy as np

class AztecBarcode:
    nparray: np.ndarray = None

    def __init__(self, nparray) -> None:
        self.nparray = nparray

    
    def get_mode()