# PYZTEC
___

A open source, python implementation of Aztec 2D barcode.


## Installation
I have not yet submitted it to pip so the only way to install this project would be by cloning (or using Download as ZIP from Github)

```shell
git clone https://github.com/ronniebasak/pyztec
cd pyztec
pip install -r requirement.txt
pip install .
```

> Note: You need to write `pip install -e .` if you want to keep editing the project


## Usage
Currently, you need a perfectly cropped and centered picture in order to be decoded, there is no correction of rotation nor reflection in any axis whatsoever. 

Reed solomon error correction is present. So a slightly corrupted image data will hopefully be recovered. There is no support for Erasure detection (which, if triggered improves error correction by upto 2x)


### Decoding

```shell
python aztec_decoder.py <image_path> <layers> [<channel>]
```

> **TIP:** to calculate the number of layers, subtract 11 from the dimention (squares) and divide by 4
> **Channel:** Which channel to the image exists in (0-R, 1-G, 2-B, 3-A)


### Encoding
```shell
python aztec_encoder.py <msg> <output_file>
```
> TIP: We've only tested PNG output files so far.

## What is this?
This project aims to add a fast, standards compliant AZTEC barcode encoding and decoding support


## Why Aztec?

It all started when I was boarding a plane and it was raining and an edge of my boarding pass got ripped. It still scanned flawlessly. This hit me, this is why airline industry uses Aztec and not QR. If the corder of a QR is ripped, the finder pattern is gone. 

Anyways, I got intrigued and I tried to find open source implentation of Aztec Codec and couldn't find any. Had to watch this YT video https://www.youtube.com/watch?v=xtlqYx6e1TE, read a patent and go through wikipedia 8 times to even get started. the IEC paper costs more than I am willing to shell on a pet project. If someone have a copy, please send me


## What is supported?

1. Lowercase and Uppercase characters
2. Digits
3. Punctionation symbols
4. **Encoding shifts:** When encoding an aztec code, after adding stuffed bits, it might be better shift to a higher layer, rather than staying in the same layer size. This is not yet supported, right now, we will remove ECC codewords to facilitate encoding

## What is **not** supported

1. FLG_N Escape sequences (ECI codes and FNC1)
2. GS1 Compliant codes
3. Binary data
4. Non ASCII character sets

## Why FULL Aztec codes aren't supported
1. They contain 1 extra ring. 
2. They contain 40 bit mode message instead of 28 bit
3. They have upto 12 bits per symbol
4. They contain a grid that needs to be erased and skipped while en/decoding


## Plans for future (in order)

### Core
1. Add Binary Data codec support
2. Add Escape sequences support (including FNC1 and GS1 symbols)
3. Add Non ASCII Charset support
4. Add Full Aztec codec support (and seamlessly transition)
5. Add mixed mode support (honestly, I have no clue why it exists)

### Utilities
1. Recognition of finder pattern from an arbitrary image
1. Add reflection/rotation detection and correction
1. Add a better API for integration