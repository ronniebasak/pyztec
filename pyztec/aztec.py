from calendar import c
from tkinter import N
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import reedsolo 
from enum import Enum

class AztecPolynomials(Enum):
    POLY_4 = 19
    POLY_6 = 67
    POLY_8 = 301
    POLY_10 = 1033
    POLY_12 = 4201


class AztecType(Enum):
    COMPACT = 0
    FULL = 1

class SpecialChars(Enum):
    PUNCT_SHIFT = "P/S"
    LOWER_LATCH = "L/L"
    MIXED_LATCH = "M/L"
    DIGIT_LATCH = "D/L"
    BINARY_SHIFT = "B/S"
    UPPER_SHIFT = "U/S"
    PUNCT_LATCH = "P/L"
    UPPER_LATCH = "U/L"
    FLG_N = "FLG(n)"
    FNC1 = "FNC1"
    ECI = "ECI"

codes = {
    "upper": [
        SpecialChars.PUNCT_SHIFT, 
        " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        SpecialChars.LOWER_LATCH, SpecialChars.MIXED_LATCH, SpecialChars.DIGIT_LATCH, SpecialChars.BINARY_SHIFT
    ],

    "lower": [
        SpecialChars.PUNCT_SHIFT,
        " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        SpecialChars.UPPER_SHIFT, SpecialChars.MIXED_LATCH, SpecialChars.DIGIT_LATCH, SpecialChars.BINARY_SHIFT
    ],

    "mixed": [
        SpecialChars.PUNCT_SHIFT,
        " ", "^A", "^B", "^C", "^D", "^E", "^F", "^G", "^H", "^I", "^J", "^K", "^L", "^M", "^[", "^\"", "^]", "^^", "^_", "@", "\\", "^", "_", "`", "|", "~", "^?",
        SpecialChars.LOWER_LATCH, SpecialChars.UPPER_LATCH, SpecialChars.PUNCT_LATCH, SpecialChars.BINARY_SHIFT
    ],

    "punct": [
        SpecialChars.FLG_N,
        "\r", "\r\n", ". ", ", ", ": ", "!", "\"","#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "[", "]", "^", "_", "{",  "}",
        SpecialChars.LOWER_LATCH, SpecialChars.UPPER_LATCH, SpecialChars.PUNCT_LATCH, SpecialChars.BINARY_SHIFT
    ],

    "digit": [
        SpecialChars.PUNCT_SHIFT, " ",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".",
        SpecialChars.UPPER_LATCH, SpecialChars.UPPER_SHIFT
    ]
}

class AztecBarcode:
    nparray: np.ndarray = None
    type: Enum = AztecType.COMPACT # default type is compact
    num_layers: int = None
    num_codewords: int = None

    def __init__(self, nparray) -> None:
        self.nparray = nparray


    def _generate_compact_mode_sequences(self) -> Tuple[int, int]:
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

    
    def _generate_compact_data_codewords(self):
        CODEWORD_SIZE = 6
        n_squares = self.nparray.shape[0]**2 
        data_squares = n_squares - 121
        skip_bits = data_squares % CODEWORD_SIZE  

        x = 0
        y = 0
        pos = 0
        
        kout = [(x,y)]


        d_rotation = ["X_IN", "Y_NIN", "X_NIN", "Y_IN"]
        direction = "X_IN"
        while True:
            if pos<6:
                if direction == "X_IN":
                    if x&1 == 0:
                        x+=1
                    if x&1 == 1:
                        x-=1
                        y+=1
                
                elif direction == "Y_NIN":
                    if y&1 == 0:
                        y-=1
                    if y&1 == 1:
                        y+=1
                        x+=1
                
                elif direction == "X_NIN":
                    if x&1 == 0:
                        x-=1
                    if x&1 == 1:
                        x+=1
                        y-=1

                elif direction == "Y_IN":
                    if y&1 == 0:
                        y+=1
                    if y&1 == 1:
                        y-=1
                        x-=1

                pos+=1


    def get_mode(self) -> Tuple[int, int]:
        assert self.type == AztecType.COMPACT, "Unsupported type"

        dimention, _ = self.nparray.shape
        center = dimention // 2
        # this is the central part of the barcode
        c_crop = self.nparray[center-5:center+6, center-5:center+6]
        
        # plt.imshow(c_crop, cmap='Greys')
        # plt.show()

        # generate all the crops
        gseq = self._generate_compact_mode_sequences()
        bit_seq = []

        # read the bit sequence
        for k in gseq:
            bit_seq.append(c_crop[k[0], k[1]])
            # c_crop[k[0], k[1]] += 2

        # convert the bit sequence to 4 bit words
        word_array = []
        for bit4 in range(0, len(bit_seq), 4 ):
            fseq = bit_seq[bit4:bit4+4]
            k = fseq[0] << 3 | fseq[1] << 2 | fseq[2] << 1 | fseq[3]
            word_array.append(k)
        
        num_layers, codewords = 0, 0
        if self.type == AztecType.COMPACT:
            # perform Reed solomon decoding for error correction
            rsc = reedsolo.RSCodec(5, nsize= len(bit_seq)//4, c_exp=4, fcr=1, prim=AztecPolynomials.POLY_4.value)
            v = rsc.decode(word_array)

            # convert the decoded words into a 16 bit integer
            mode_msg = 0
            for byte in v[0]:
                mode_msg = (mode_msg << 4) | byte
            
            # compute layer count and codeword from the mode message
            num_layers = ( (mode_msg >> 6) & 0b11 ) + 1
            codewords = (mode_msg & 0b111111) + 1
            self.num_layers = num_layers
            self.num_codewords = codewords

        return num_layers, codewords

    
    def decode(self):
        assert self.type == AztecType.COMPACT, "Unsupported type"
        CODEWORD_SIZE = 6
        if not self.num_layers and self.nparray.any():
            self.get_mode()

        n_squares = self.nparray.shape[0]**2 
        data_squares = n_squares - 121
        skip_bits = data_squares % CODEWORD_SIZE  