from calendar import c
from tkinter import N
from tkinter.messagebox import RETRY
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

class AztecBarcodeCompact:
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

    
    # layer is 0 indexed
    def _get_layer_boundary(self, layer, direction):
        total_layers =  self.num_layers #(self.nparray.shape[0] - 15) // 4
        center = self.nparray.shape[0] // 2
        # print(center, self.nparray.shape[0])
        lsize = layer * 4 + 15
        loff = 2*(self.num_layers - layer - 1)

        x,y, xn, yn, next_layer = None, None, None, None, layer
        if direction == "X_IN":
            # calculate boundary condition
            x = center - ( 7 + 2*layer) 
            y = lsize - 2 + loff

            # calculate next start of codeword
            xn = x
            yn = y+1

        elif direction == "Y_NIN":
            x = lsize - 2 + loff
            y = center + ( 7 + 2*layer)

            xn = x+1
            yn = y

        elif direction == "X_NIN":
            x = lsize - 1 + loff
            y = center - ( 7 + 2*layer) + 1

            xn = x
            yn = y-1
        
        elif direction == "Y_IN":
            x = center - ( 7 + 2*layer) + 1
            y = center - ( 7 + 2*layer)

            xn = x+1
            yn = y+2

            next_layer = layer-1 if layer > 0 else None
        return x,y, xn, yn, next_layer 
            

    
    def _generate_compact_data_codewords(self, CODEWORD_SIZE: int):
        assert self.type == AztecType.COMPACT, "Unsupported type"
        n_squares = self.nparray.shape[0]**2 
        data_squares = n_squares - 121
        skip_bits = data_squares % CODEWORD_SIZE  

        x = 0
        y = 0
        pos = 0
        
        kout = [(x,y)]
        layer = (self.nparray.shape[0] - 15) // 4

        d_rotation = ["X_IN", "Y_NIN", "X_NIN", "Y_IN"]
        d_ind = 0
        direction = d_rotation[d_ind]
        x_match, y_match, x_next, y_next, layer_next = self._get_layer_boundary(layer, direction)
        # codecount = 0

        if skip_bits == 0:
            pos += 1
            yield x,y

        while True:
            if skip_bits:
                if pos&1 == 0:
                    x+=1
                else:
                    x-=1
                    y+=1
                pos += 1
                skip_bits -= 1
                if skip_bits == 0:
                    pos = 1
                    yield x,y

            elif pos< CODEWORD_SIZE:
                if direction == "X_IN": ## READING INWARDS x-direction while going down
                    if x&1 == 0:
                        x+=1
                    elif x&1 == 1:
                        x-=1
                        y+=1
                
                elif direction == "Y_NIN": # READING INWARDS y-direction while going right
                    if y&1 == 0:
                        y-=1
                    elif y&1 == 1:
                        y+=1
                        x+=1
                
                elif direction == "X_NIN": # READING INWARDS x-direction while going up
                    if x&1 == 0:
                        x-=1
                    elif x&1 == 1:
                        x+=1
                        y-=1

                elif direction == "Y_IN": # READING INWARDS y-direction while going left
                    if y&1 == 0:
                        y+=1
                    elif y&1 == 1:
                        y-=1
                        x-=1

                if pos&1==0:
                    # check if we need to switch direction
                    if x == x_match and y == y_match:
                        d_ind = (d_ind + 1 ) % 4
                        direction = d_rotation[d_ind]
                        x,y = x_next, y_next
                        layer = layer_next
                        if layer is None:
                            break

                        x_match, y_match, x_next, y_next, layer_next = self._get_layer_boundary(layer, direction)


                pos+=1
                if pos == CODEWORD_SIZE:
                    pos = 0
                    # codecount += 1
                yield x,y


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
        if not self.num_layers and self.nparray.any():
            self.get_mode()

        CODEWORD_SIZE = 6 if self.num_layers < 3 else 8 # compute codeword size here
        # n_squares = self.nparray.shape[0]**2 
        # data_squares = n_squares - 121

        codewords = self._get_codewords(CODEWORD_SIZE)
        
        bitstring = self._get_bit_string(codewords, CODEWORD_SIZE)
        # print(bitstring)

        ind = 0
        mode="upper"
        inc = 5
        shift = False
        pmode = None
        output = []
        
        while ind<len(bitstring):
            if mode != "digit":
                bits = bitstring[ind: ind+5]
                if len(bitstring) - ind <= 5:
                    break
                inc = 5
            else:
                bits = bitstring[ind: ind+4]
                if len(bitstring) - ind <= 4:
                    break
                inc = 4

            # print("bits",bits)

            data = int(bits, 2)
            # print("data",data)
            decoded_char = codes[mode][data]
            # print(decoded_char)

            if shift:
                shift = False
                mode = pmode
                pmode = None
            if decoded_char == SpecialChars.LOWER_LATCH:
                mode = "lower"
            elif decoded_char == SpecialChars.UPPER_LATCH:
                mode = "upper"
            elif decoded_char == SpecialChars.UPPER_SHIFT:
                shift = True
                pmode = mode
                mode = "upper"
            elif decoded_char == SpecialChars.DIGIT_LATCH:
                mode = "digit"
            elif decoded_char == SpecialChars.PUNCT_LATCH:
                mode = "punct"
            elif decoded_char == SpecialChars.PUNCT_SHIFT:
                shift = True
                pmode = mode
                mode = "punct"


            else:
                output.append(decoded_char)
            ind += inc
        # print(output)
        if output[-1] == SpecialChars.BINARY_SHIFT or output[-1] == SpecialChars.UPPER_SHIFT or output[-1] == SpecialChars.UPPER_LATCH:
            output = output[:-1]
        return output



    def _get_bit_string(self, codewords, CODEWORD_SIZE: int):
        GF_SIZE = 0
        PRIME = 0

        # REED SOLOMON CORRECTION 
        if self.num_layers < 3:
            GF_SIZE = 6
            PRIME = AztecPolynomials.POLY_6.value
        elif self.num_layers < 9:
            GF_SIZE = 8
            PRIME = AztecPolynomials.POLY_8.value
        elif self.num_layers < 23:
            GF_SIZE = 10
            PRIME = AztecPolynomials.POLY_10.value
        else:
            GF_SIZE = 12
            PRIME = AztecPolynomials.POLY_12.value

        rsc = reedsolo.RSCodec(5, nsize= len(codewords), c_exp=GF_SIZE, fcr=1, prim=PRIME)
        v = rsc.decode(codewords)
        useful_codewords = list(v[0][:self.num_codewords])
        # useful_codewords = codewords[:self.num_codewords]
        # print("DECODED WORDS", list(v[0]))
        MAX_BITS = (1<<CODEWORD_SIZE) - 1 # all ones
        MAX_MSBITS = MAX_BITS >> 1 # all msb ones

        bitstring = ""
        for codeword in useful_codewords:
            # unstuff bits
            # print("CODEWORD", codeword, "{:6b}".format(codeword).replace(" ", "0"))
            if codeword == 0 or codeword == MAX_BITS:
                raise ValueError("All bits are the same, erasure detected")

            bits = ""
            if codeword == 1:
                bits = "0" * (CODEWORD_SIZE -1)

            elif codeword &1 ==0 and (codeword>>1) == MAX_MSBITS:
                bits = "1"* (CODEWORD_SIZE -1)
            else:
                bits = "{:b}".format(codeword)
                bits = "0"* (CODEWORD_SIZE - len(bits)) + bits

            # print("DECODED BITS", bits)
            bitstring += bits
        # print("BIT STRING::: ", bitstring)
        return bitstring


    def _get_codewords(self, CODEWORD_SIZE):
        codewords = []
        debug = ""

        pos, codecount = 0,0
        acc = 0
        for x,y in self._generate_compact_data_codewords(CODEWORD_SIZE):
            bit = self.nparray[y,x]
            # print(x,y, bit)
            acc = acc << 1 | bit
            # debug += str(bit)
            pos += 1
            if pos == CODEWORD_SIZE:
                pos = 0
                codecount += 1
                codewords.append(acc)
                acc = 0
                debug += " "
        # print(debug)
        return codewords

