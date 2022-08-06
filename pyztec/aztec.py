from calendar import c
from tkinter import N
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import reedsolo 


def prod(x, y, log, alog, gf):
    """ Product x times y """
    if not x or not y:
        return 0
    return alog[(log[x] + log[y]) % (gf - 1)]


def reed_solomon(ind, nd, nc, gf, pp):
    """ Calculate error correction codewords
    
    Algorithm is based on Aztec Code bar code symbology specification from
    GOST-R-ISO-MEK-24778-2010 (Russian)
    Takes ``nd`` data codeword values in ``ind`` and adds on ``nc`` check
    codewords, all within GF(gf) where ``gf`` is a power of 2 and ``pp``
    is the value of its prime modulus polynomial.

    :param ind: data codewords (in/out param)
    :param nd: number of data codewords
    :param nc: number of error correction codewords
    :param gf: Galois Field order
    :param pp: prime modulus polynomial value
    """

    wd = ([*ind] + [0] * nc)[:nd + nc]
    # generate log and anti log tables
    log = {0: 1 - gf}
    alog = {0: 1}
    for i in range(1, gf):
        alog[i] = alog[i - 1] * 2
        if alog[i] >= gf:
            alog[i] ^= pp
        log[alog[i]] = i
    # generate polynomial coeffs
    c = {0: 1}
    for i in range(1, nc + 1):
        c[i] = 0
    for i in range(1, nc + 1):
        c[i] = c[i - 1]
        for j in range(i - 1, 0, -1):
            c[j] = c[j - 1] ^ prod(c[j], alog[i], log, alog, gf)
        c[0] = prod(c[0], alog[i], log, alog, gf)
    # generate codewords
    for i in range(nd, nd + nc):
        wd[i] = 0
    for i in range(nd):
        assert 0 <= wd[i] < gf
        k = wd[nd] ^ wd[i]
        for j in range(nc):
            wd[nd + j] = prod(k, c[nc - j - 1], log, alog, gf)
            if j < nc - 1:
                wd[nd + j] ^= wd[nd + j + 1]
    return wd

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

        cwd= reed_solomon([0,5], 2, 5, 16, 19)
        print(cwd)



        # plt.imshow(c_crop, cmap='Greys')
        # plt.show()