from typing import Any, List, Tuple
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
        "\r", "\r\n", ". ", ", ", ": ", "!", "\"","#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "[", "]", "{",  "}",
        SpecialChars.UPPER_LATCH
    ],

    "digit": [
        SpecialChars.PUNCT_SHIFT, " ",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".",
        SpecialChars.UPPER_LATCH, SpecialChars.UPPER_SHIFT
    ]
}


# each tuple contains the number of bits and the number of symbols
# followed by number of skipped bits 
codeword_size_map = {
    1: (6, 17, 2),
    2: (6, 40, 0),
    3: (8, 51, 0),
    4: (8, 76, 0)
}


class AztecCodecModes(Enum):
    READ: int = 0
    WRITE: int = 1


class AztecBarcodeCompact:
    nparray: np.ndarray = None
    type: Enum = AztecType.COMPACT # default type is compact
    num_layers: int = None
    num_codewords: int = None
    codec_mode: AztecCodecModes = AztecCodecModes.READ

    def __init__(self, nparray, codec_mode: AztecCodecModes = AztecCodecModes.READ) -> None:
        self.nparray = nparray
        if codec_mode == AztecCodecModes.WRITE:
            self.nparray = np.zeros((11,11))

        self.codec_mode = codec_mode

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
            

    
    def _read_compact_data_codewords(self, CODEWORD_SIZE: int):
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
        for x,y in self._read_compact_data_codewords(CODEWORD_SIZE):
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



    def _get_char_class(self, char: str) -> str:
        to_return = []
        subarr_punct = ["{", "}"]
        subarr2_punct = ["\r\n", ". ", ", ", ": "]
        subarr_mixed = ["@", "\\", "^", "_", "`", "|", "~"]
        
        if len(char) == 1:
            # upper
            if 65 <= ord(char) <= 90:
                to_return.append(("upper", 2 + (ord(char) - 65) ))

            # Lower
            elif 97 <= ord(char) <= 122:
                to_return.append(("lower", 2 + (ord(char) - 97) ))

            # Punctuation
            elif 33 <= ord(char) <= 47:
                if char == ",":
                    to_return.append(("digit", 12))
                elif char == ".":
                    to_return.append(("digit", 13))
                to_return.append(("punct", 6 + (ord(char) - 33) ))

            elif 58 <= ord(char) <= 63:
                to_return.append(("punct", 21 + (ord(char) -  58)) )
            elif 93 <= ord(char) <= 95:
                to_return.append(("punct", 28 + (ord(char) - 93) ))

            elif char == "\r":
                    to_return.append(("punct", 1))
            elif char == "[":
                to_return.append(("punct", 27))

            elif char in subarr_punct:
                ind = subarr_punct.index(char)
                to_return.append(("punct", 30+ind))

            elif char == " ":
                to_return += [("upper", 1), ("lower", 1), ("mixed", 1), ("digit", 1)]

            elif char in subarr_mixed:
                ind = subarr_mixed.index(char)
                to_return.append(("punct", 20+ind))
            
            elif 48 <= ord(char) <= 57:
                to_return.append(("digit", 2 + ord(char) - 48))
                
                
        elif len(char) == 2:
            if char in subarr2_punct:
                ind = subarr2_punct.index(char)
                to_return.append(("punct", 2 + ind))
        
        return to_return

    
    def _calculate_mode_seq(self, mode, emode, ecode, char_ind, input_string, cskip):
        seq = []
        skip = cskip
        # if same mode, then continue
        if mode == emode:
            if mode == "digit":
                seq += [(ecode, 4)]
            else:
                seq += [ecode]

        elif emode == "lower":
            if mode == "upper" or mode == "mixed":
                seq += [codes[mode].index(SpecialChars.LOWER_LATCH), ecode]
            elif mode == "digit" or mode == "punct":
                seq += [codes[mode].index(SpecialChars.UPPER_LATCH), codes["upper"].index(SpecialChars.LOWER_LATCH), ecode]

            mode = emode

        elif emode == "upper":
            if mode == "lower":
                # when shifting from lower to upper,
                # we need to check if at least 3 chars are upper, then latch else shift
                if char_ind < len(input_string) -2:

                    k1 = self._get_char_class(input_string[char_ind+1])
                    ncx, ncxc = k1[0]

                    if ncx == "upper":
                        seq += [codes[mode].index(SpecialChars.DIGIT_LATCH), (codes["digit"].index(SpecialChars.UPPER_LATCH), 4), ecode,  ncxc]
                        skip = 2
                        mode = emode
                    else:
                        seq += [codes[mode].index(SpecialChars.UPPER_SHIFT), ecode]
            
                else:
                    seq += [codes[mode].index(SpecialChars.UPPER_SHIFT), ecode]
        
            elif mode == "punct" or mode == "mixed":
                seq += [codes[mode].index(SpecialChars.UPPER_LATCH), ecode]
                mode = emode
            
            # if next character is digit then we shift else we latch
            elif mode == "digit":
                k1 = self._get_char_class(input_string[char_ind+1])
                ncx, ncxc = k1[0]

                if ncx == "digit":
                    seq += [(codes[mode].index(SpecialChars.UPPER_SHIFT), 4), ecode]
                else:
                    seq += [(codes[mode].index(SpecialChars.UPPER_LATCH), 4), ecode]
                    mode = emode


        elif emode == "punct":
            if mode == "digit":
                seq += [(0, 4), ecode]
            else:
                seq += [0, ecode]
            mode = emode

        
        elif emode == "digit":
            if mode == "upper" or mode == "lower":
                seq += [codes[mode].index(SpecialChars.DIGIT_LATCH), (ecode, 4)]
            elif mode == "punct" or mode == "mixed":
                seq += [codes[mode].index(SpecialChars.UPPER_LATCH), codes["upper"].index(SpecialChars.DIGIT_LATCH), (ecode, 4)]
            mode = emode
        
        elif emode == "mixed":
            if mode == "upper" or mode =="lower":
                seq += [codes[mode].index(SpecialChars.MIXED_LATCH), ecode]
            
            elif mode == "digit" or mode == "punct":
                seq += [codes[mode].index(SpecialChars.UPPER_LATCH), codes["upper"].index(SpecialChars.MIXED_LATCH), ecode]
            
            mode = emode

        return seq, skip, mode


    def _convert_input_string_to_seq(self, input_string: str) -> List[Any]:
        prev_char = None
        prev_mode = None
        mode = "upper"
        seq = []

        char_ind = 0
        while char_ind < len(input_string):
            skip = 1
            char = input_string[char_ind]

            available_modes = []
            
            if char == "\r" and char_ind < len(input_string) - 1 and input_string[char_ind+1] == "\n":
                available_modes = self._get_char_class(input_string[char_ind:char_ind+2])
                skip = 2
            
            elif char in [".",",", ":"] and char_ind < len(input_string) - 1 and input_string[char_ind+1] == " ":
                available_modes = self._get_char_class(input_string[char_ind:char_ind+2])
                skip = 2
            
            else:
                available_modes = self._get_char_class(char)

            if len(available_modes) == 1:
                emode, ecode = available_modes[0]
                _seq, _skip, _mode = self._calculate_mode_seq(mode, emode, ecode, char_ind, input_string, skip)
                seq += _seq
                skip = _skip
                mode = _mode


            else:
                modi = -1
                for im in range(len(available_modes)):
                    if available_modes[im][0] == mode:
                        modi = im
                        break
                
                if modi >= 0:
                    emode, ecode = available_modes[modi]
                    _seq, _skip, _mode = self._calculate_mode_seq(mode, emode, ecode, char_ind, input_string, skip)
                    seq += _seq
                    skip = _skip
                    mode = _mode
                else:
                    # can be optimised later
                    emode, ecode = available_modes[0]
                    _seq, _skip, _mode = self._calculate_mode_seq(mode, emode, ecode, char_ind, input_string, skip)
                    seq += _seq
                    skip = _skip
                    mode = _mode

            char_ind += skip
        return seq


    def _convert_charray_bitstring(self, charray: List[Any]) -> str:
        bitstring = ""
        for k in charray:
            item = k
            bits = 5
            if type(k) == tuple:
                item, bits = k

            _str = "{:b}".format(item)
            prepad = "0" * (bits - len(_str))
            _str = prepad + _str
            bitstring += _str
        return bitstring


    def _convert_bitstring_bitstuff_pad(self, bitstring: str) -> str:
        bitlength = len(bitstring)
        k=0
        while k < bitlength:
            # add pad bits at the end
            if len(bitstring) - k < 6:
                pads = "1" * (5 - (len(bitstring) - k))

                if bitstring[k:k+6] + pads == "11111":
                    pads += "0"
                else:
                    pads += "1"
                bitstring += pads

            if bitstring[k:k+6] == "111111":
                bitstring = bitstring[:k] + "111110" + bitstring[k+5:]
                bitlength += 1

            if bitstring[k:k+6] == "000000":
                bitstring = bitstring[:k] + "000001" + bitstring[k+5:]
                bitlength += 1

            k+=6
        
        # 1 codeword worth of padding
        bitstring += "111110"
        return bitstring
        

    def _compute_codewords_from_bitstring(self, bitstring: str, CODEWORD_SIZE: int = 6) -> List[int]:
        codewords = []
        assert len(bitstring) % CODEWORD_SIZE == 0, "Bitstring must be a multiple of 6"
    
        for i in range(0, len(bitstring), CODEWORD_SIZE):
            subs = bitstring[i:i+CODEWORD_SIZE]
            subs_int = int(subs, 2)
            codewords.append(subs_int)
        return codewords


    def _calculate_boundaries(self, ecc_level):
        if ecc_level == 1:
            ECC_FACTOR = 1.23
            ECC_OFFSET = 3

            boundaries = {}
            for c in range(1,5):
                ss, cw, _ = codeword_size_map[c]

                boundaries[c] = ss*( (cw - ECC_OFFSET)//ECC_OFFSET )


    # returns layer size and total supported codewords
    def _compute_layer_size(self, bitstring, ecc_level = 0) -> Tuple[int, int]:
        symbol_size, codewords, skip = 0, 0, 0

        if len(bitstring) <= 60:
            symbol_size, codewords, skip = codeword_size_map[1]
        elif len(bitstring) <= 180:
            symbol_size, codewords, skip = codeword_size_map[2]
            return codeword_size_map[2]
        elif len(bitstring) <= 304:
            return codeword_size_map[3]



    def _reedsolo_enc(self, codewords):
        ...


    def _get_nparray_from_codewords(self, codewords):
        ... 


    def encode(self, input_string: str | List[Any]):
        self.codec_mode = AztecCodecModes.WRITE

        # convert string into character array, includes escape sequences and special characters.
        char_arr = self._convert_input_string_to_seq(input_string)

        # convert the character array into binary string
        bitstring = self._convert_charray_bitstring(char_arr)

        # calculate layer size required
        lsize = self._compute_layer_size(bitstring)
        
        # add stuffing bits and padding bits if necessary
        bitstring = self._convert_bitstring_bitstuff_pad(bitstring)

        # split the string into 6 bit codewords
        codewords = self._compute_codewords_from_bitstring(bitstring); # array of codewords that needs to be encoded
        # add reed-solomon codewords
        codewords = self._reedsolo_enc(codewords)

        # encode into the np array
        nparray = self._get_nparray_from_codewords(codewords)

        # return np array
        return nparray



