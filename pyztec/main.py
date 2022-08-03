from turtle import shape
import numpy as np
import math


def draw_square(arr, ring):
    assert arr.shape[0] == arr.shape[1] and arr.shape[0] &1 == 1, "Array must be a square of odd numbers"
    MIDPOINT = arr.shape[0] //2
    assert ring <= MIDPOINT and ring >= 0, "Ring size must be within limits of the array"
    
    high = MIDPOINT + ring
    low = MIDPOINT - ring

    for q in range(low, high+1):
        arr[high][q] = 1
        arr[low][q] = 1
        arr[q][high] = 1
        arr[q][low] = 1

COMPACT_AZTEC_CODE = np.zeros(shape=(11,11), dtype=int, order='C')
draw_square(COMPACT_AZTEC_CODE, 0)
draw_square(COMPACT_AZTEC_CODE, 2)
draw_square(COMPACT_AZTEC_CODE, 4)


# constant = 


# class PyZtec:
#     def __init__(self):
#         self.array = np.array([])

#     def run(self):
#         pass

#     def __del__(self):
#         pass



print(COMPACT_AZTEC_CODE)