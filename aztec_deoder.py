import sys
import imageio.v3 as imageio
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import io
import base64
from pyztec.aztec import AztecBarcodeCompact


# usage python img2nparr.py <filename> <layers> [<channel>]
# this file generates a Base64 Serialised NumPy array

assert len(sys.argv) > 2, "Too few arguments"
assert os.path.exists(sys.argv[1]), "Can't find the file, CWD: %s" % (os.getcwd())

assert 0 <= int(sys.argv[2]) <= 4, "Layers must be a positive int between 0 and 4"
CHANNEL = int(sys.argv[3]) if len(sys.argv) > 3 else 2 # the third argument is channel

layers = int(sys.argv[2])
dimension = layers*4 + 11

image = imageio.imread(sys.argv[1]) # Open file for reading
print(sys.argv[1])
if sys.argv[1].endswith(".gif"):
    image = image[0] # only the first frame
# plt.imshow(image)



image_alpha = image[:,:, CHANNEL]; # mentioned channel
# plt.imshow(image_alpha, cmap="Greys")
pil_img = Image.fromarray(image_alpha).resize((dimension, dimension)).convert('1')
# plt.imshow(pil_img, cmap="Greys")
nparr = np.array(pil_img)

if CHANNEL != 3: # in alpha channel, we get correct inversion else bits are flipped
    nparr = ~nparr
# plt.imshow(nparr, cmap="Greys")
# plt.show()

## THIS SECTION GENERATES DEBUG SEQUENCE
# memfile = io.BytesIO()
# np.save(memfile, nparr*1)
# data = memfile.getvalue()
# print(base64.b64encode(data).decode('latin-1`'))

# THIS SECTION ATTEMPTS TO DECODE
aztec = AztecBarcodeCompact(nparr)
v = aztec.decode()
print(v)
print("".join(v))