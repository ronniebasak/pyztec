from re import A
import pytest
import numpy as np
from pyztec.aztec import AztecBarcodeCompact



class TestEncode:

    def test__convert_input_string_to_seq(self):
        aztec = AztecBarcodeCompact(np.zeros((11,11))) 
        test_cases = [
            # Testing upper case single character
            {
                "input": "A",
                "output": [2]
            },

            # Upper case multiple characters
            {
                "input": "ABCDE",
                "output": [2,3,4,5,6]
            },

            # Upper to lower case
            {
                "input": "Abcde",
                "output": [2, 28, 3,4,5,6]
            },

            # Lower to upper tiny 
            {
                "input": "aBC",
                "output": [28, 2, 28, 3, 28, 4]
            },

            # Lower to upper short
            {
                "input": "aBCD",
                "output": [28, 2, 30, (14, 4), 3, 4, 5]
            },

            # Lower to upper latch
            {
                "input": "aBCDEF",
                "output": [28, 2, 30, (14, 4), 3,4,5,6, 7]
            },

            # Checking space handling
            {
                "input": "Hello World",
                "output": [9, 28, 6, 13, 13, 16, 1, 28, 24, 16, 19, 13, 5]
            },

            # Checking upper case space handling
            {
                "input": "HELLO WORLD",
                "output": [9, 6, 13, 13, 16, 1, 24, 16, 19, 13, 5]
            },

            # Checking punct handling
            {
                "input": "Hello,World!",
                "output": [9, 28, 6, 13, 13, 16, 30, (12, 4), (14, 4), 24, 28, 16, 19, 13, 5, 0, 6]
            },
 
            # Checking comma space condensation handling typical
            {
                "input": "Hello, World!",
                "output": [9, 28, 6, 13, 13, 16, 0, 4, 31, 24, 28, 16, 19, 13, 5, 0, 6]
            },
 
            # Checking comma space condensation handling typical
            {
                "input": "Hello. World!",
                "output": [9, 28, 6, 13, 13, 16, 0, 3, 31, 24, 28, 16, 19, 13, 5, 0, 6]
            },
 
            # Checking colon space condensation handling typical
            {
                "input": "Hello: World!",
                "output": [9, 28, 6, 13, 13, 16, 0, 5, 31, 24, 28, 16, 19, 13, 5, 0, 6]
            },

            # Checking comma space condensation handling lowercase
            {
                "input": "hello, world!",
                "output": [28, 9, 6, 13, 13, 16, 0, 4, 31, 28, 24, 16, 19, 13, 5, 0, 6]
            },

            # Checking period space condensation handling lowercase
            {
                "input": "hello. world!",
                "output": [28, 9, 6, 13, 13, 16, 0, 3, 31, 28, 24, 16, 19, 13, 5, 0, 6]
            },

            # Checking colon space condensation handling lowercase
            {
                "input": "hello: world!",
                "output": [28, 9, 6, 13, 13, 16, 0, 5, 31, 28, 24, 16, 19, 13, 5, 0, 6]
            },

            # Checking CR
            {
                "input": "\r",
                "output": [0, 1]
            },

            # Checking CR LF
            {
                "input": "\r\n",
                "output": [0, 2]
            },

            # Checking CR LF in string
            {
                "input": "Hey\r\nThere",
                "output": [9, 28, 6, 26, 0, 2, 31, 21, 28, 9, 6, 19, 6]
            },

            # Checking digits in a string
            {
                "input": "1",
                "output": [30, (3, 4)]
            },

            # Checking digits in a string
            {
                "input": "12345",
                "output": [30, (3, 4), (4,4), (5,4), (6,4), (7,4)]
            },

            # Checking digits in a string
            {
                "input": "A12345",
                "output": [2, 30, (3, 4), (4,4), (5,4), (6,4), (7,4)]
            },

            # Checking digits in a string
            {
                "input": "Ai12345",
                "output": [2, 28, 10, 30, (3, 4), (4,4), (5,4), (6,4), (7,4)]
            },

            # Checking digits and spaces in a string
            {
                "input": "Ai 12345",
                "output": [2, 28, 10, 1, 30, (3, 4), (4,4), (5,4), (6,4), (7,4)]
            },

            # Checking digits and spaces in a string
            {
                "input": "Ai 123 45",
                "output": [2, 28, 10, 1, 30, (3, 4), (4,4), (5,4), (1, 4), (6,4), (7,4)]
            },

            # Checking digits and symbiols in a string
            {
                "input": "Ai 12,3.45",
                "output": [2, 28, 10, 1, 30, (3, 4), (4,4), (12, 4),(5,4), (13, 4), (6,4), (7,4)]
            },
        ]


        for test_case in test_cases:
            assert aztec._convert_input_string_to_seq(test_case["input"]) == test_case["output"], "Failed Test Case: {}".format(test_case["input"])


    def test__convert_charray_bitstring(self):
        aztec = AztecBarcodeCompact(np.zeros((11,11))) 
        test_cases = [
            # Testing upper case single character
            {
                "input": [2],
                "output": "00010"
            },

            # Upper case multiple characters
            {
                "input": [2,3,4,5,6],
                "output": "0001000011001000010100110"
            },

            # mixed
            {
                "input": [2,28,4,5,6],
                "output": "0001011100001000010100110"
            },
            # digits
            {
                "input": [2, 3, 30, (12,4)],
                "output": "0001000011111101100"
            },
        ]

        for test_case in test_cases:
            assert aztec._convert_charray_bitstring(test_case["input"]) == test_case["output"], "Failed Test Case: {}".format(test_case["input"])
        

    def test__convert_bitstring_bitstuff_pad(self):
        aztec = AztecBarcodeCompact(np.zeros((11,11))) 
        test_cases = [
            # 4 bits
            {
                "input": ("0001", 6),
                "output": "000111 111110"
            },
            # 5 bits
            {
                "input": ("00010", 6),
                "output": "000101 111110"
            },

            # 6 bits
            {
                "input": ("000100", 6),
                "output": "000100 111110"
            },

            # 6 1st bit
            {
                "input": ("111111", 6),
                "output": "111110 111110 111110"
            },

            # 5 1st bit and a 0
            {
                "input": ("111110", 6),
                "output": "111110 111110"
            },

            # 6 0s
            {
                "input": ("000000", 6),
                "output": "000001 011111 111110"
            },


            # random bits 
            {
                "input": ("000111 000111", 6),
                "output": "000111 000111 111110"
            },

            # random bits with all 1s
            {
                "input": ( "000111 111111 000111", 6),
                "output": "000111 111110 100011 111110 111110"
            },

            # 8 bit test
            {
                "input": ( "000101", 8),
                "output": "00010111 11111110"
            },

            # 8 bit test all 0s
            {
                "input": ( "00000000", 8),
                "output": "00000001 01111111 11111110"
            },

            # 8 bit test all 1s
            {
                "input": ( "11111111", 8),
                "output": "11111110 11111110 11111110"
            },

            # 8 bit test all 1 in btween
            {
                "input": ( "00010010 11111111", 8),
                "output": "00010010 11111110 11111110 11111110"
            },
        ]

        for test_case in test_cases:
            assert aztec._convert_bitstring_bitstuff_pad(*test_case["input"]) == test_case["output"].replace(" ", ""), "Failed Test Case: {}".format(test_case["input"])
        

    def test__compute_codewords_from_bitstring(self):
        aztec = AztecBarcodeCompact(np.zeros((11,11))) 
        test_cases = [
            {
                "input": ("010100111110", 6),
                "output": [20, 62]
            },
            {
                "input": ("000000111110", 6),
                "output": [0, 62]
            },
            {
                "input": ("010100000001111100", 6),
                "output": [20, 1, 60]
            },
            {
                "input": ("00001000", 8),
                "output": [8]
            },
            {
                "input": ("0000100010101010", 8),
                "output": [8, 170]
            },
        ]

        for test_case in test_cases:
            assert aztec._compute_codewords_from_bitstring(*test_case["input"]) == test_case["output"], "Failed Test Case: {}".format(test_case["input"])


    def test__reedsolo_enc(self):
        aztec = AztecBarcodeCompact(np.zeros((11,11))) 
        test_cases = [
            {
                "input": ([19, 48, 51, 22, 48, 62], 17, 1),
                "output": [19, 48, 51, 22, 48, 62, 26, 63, 21, 32, 50, 45, 16, 14, 46, 12, 14]
            },

            {
                "input": ([175, 18, 170, 5, 84, 8, 158, 19, 100, 78, 139, 76, 30, 11, 117, 49, 3, 77, 56, 112, 104, 79, 9, 135, 152, 83, 21, 19, 21, 66, 4, 254], 51, 3),
                "output": [175, 18, 170, 5, 84, 8, 158, 19, 100, 78, 139, 76, 30, 11, 117, 49, 3, 77, 56, 112, 104, 79, 9, 135, 152, 83, 21, 19, 21, 66, 4, 254, 83, 143, 82, 140, 129, 100, 253, 56, 114, 233, 189, 155, 98, 226, 151, 139, 53, 211, 64]
            },

            {
                "input": ([57, 35, 6, 1, 9, 56, 33, 13, 41, 17, 34, 38, 18, 1, 41, 42, 1, 19, 9, 1, 59], 40, 2),
                "output": [57, 35, 6, 1, 9, 56, 33, 13, 41, 17, 34, 38, 18, 1, 41, 42, 1, 19, 9, 1, 59, 24, 10, 17, 13, 57, 19, 23, 15, 19, 58, 62, 55, 8, 1, 58, 44, 58, 31, 57]
            },
        ]

        for test_case in test_cases:
            assert aztec._reedsolo_enc(*test_case["input"]) == test_case["output"],  "Failed Reed Solomon Case: {}".format(test_case["input"])
