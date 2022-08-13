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
                "input": "0001",
                "output": "000111111110"
            },
            # 5 bits
            {
                "input": "00010",
                "output": "000101111110"
            },

            # 6 bits
            {
                "input": "000100",
                "output": "000100111110"
            },

            # 6 1st bit
            {
                "input": "111111",
                "output": "111110111110111110"
            },

            # 5 1st bit and a 0
            {
                "input": "111110",
                "output": "111110111110"
            },

            # 6 0s
            {
                "input": "000000",
                "output": "000001011111111110"
            },


            # random bits 
            {
                "input": "000111000111",
                "output": "000111000111111110"
            },

            # random bits with all 1s
            {
                "input":  "000111111111000111",
                "output": "000111111110100011111110111110"
            },
        ]

        for test_case in test_cases:
            assert aztec._convert_bitstring_bitstuff_pad(test_case["input"]) == test_case["output"], "Failed Test Case: {}".format(test_case["input"])
        

    def test__compute_codewords_from_bitstring(self):
        test_cases = [
            {
                "input": "010100111110",
                "output": [20, 30]
            },
            {
                "input": "000000111110",
                "output": [0, 30]
            },
            {
                "input": "01010000000111110",
                "output": [20, 0, 30]
            },
            {
                "input": "0000",
                "output": [20, 30]
            },
        ]