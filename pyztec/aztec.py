from calendar import c
from tkinter import N
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import reedsolo 


class AztecBarcode:
    nparray: np.ndarray = None

    def __init__(self, nparray) -> None:
        self.nparray = nparray


    def _generate_sequences(self) -> Tuple[int, int]:
        xseq = [1, 0, -1, 0]
        yseq = [0, 1, 0, -1]
        xset = [2, 10, 8, 0]
        yset = [0, 2, 10, 8]
        
        dim = 0
        count = 0

        x = xset[dim]
        y = yset[dim]
        xdir = xseq[dim]
        ydir = yseq[dim]

        while True:
            if count != 0:
                x += xdir
                y += ydir
            if count == 7:
                count = 0
                dim += 1
                if dim == 4:
                    break # we're done, stop generating sequences
                xdir = xseq[dim]
                ydir = yseq[dim]
                x = xset[dim]
                y = yset[dim]
            count += 1
            yield y, x


    def get_mode(self) -> str:
        dimention, _ = self.nparray.shape
        center = dimention // 2
        # this is the central part of the barcode
        c_crop = self.nparray[center-5:center+6, center-5:center+6]
        c_center = 5 # as of now we're only looking at compact AZTECs

        iter = self._generate_sequences()

        bit_seq = []
        for k in iter:
            bit_seq.append(c_crop[k[0], k[1]])
            # c_crop[k[0], k[1]] += 2

        bit_array = []
        for bit4 in range(0, len(bit_seq), 4 ):
            fseq = bit_seq[bit4:bit4+4]
            k = fseq[0] << 3 | fseq[1] << 2 | fseq[2] << 1 | fseq[3]
            bit_array.append(k)

        print("".join(map(str, bit_seq)))
        print(bit_array)

        rsc = reedsolo.RSCodec(5, nsize=7, c_exp=4, fcr=1, prim=19)
        v = rsc.encode(bytearray([0,5]))
        print(list(v))



        # plt.imshow(c_crop, cmap='Greys')
        # plt.show()