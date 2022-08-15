from PIL import Image
import numpy as np
import sys
from pyztec.aztec import AztecBarcodeCompact
import matplotlib.pyplot as plt

# usage `aztec_encoder.py <message> <outfile>`
assert len(sys.argv) > 2, "Too few arguments"

msg = sys.argv[1]
outfile = sys.argv[2]

abc = AztecBarcodeCompact(np.zeros((11,11)))
nparr: np.ndarray = abc.encode(msg)
nparr = nparr.astype('bool')
nparr = (~nparr)*255
nparr = nparr.astype('uint8')


# plt.imshow(nparr, cmap="gray")
dim = nparr.shape[0]

img = Image.fromarray(nparr, "L")

img = img.resize((dim*10, dim*10), Image.NEAREST)
plt.imshow(img)
plt.show()
img.save(outfile)