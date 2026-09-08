import struct
import numpy as np
def readMrcSizeApix(mrcFileName):
    with open(mrcFileName, "rb") as mrcFile:
        imageSizeX = int(struct.unpack('i', mrcFile.read(4))[0])
        imageSizeY = int(struct.unpack('i', mrcFile.read(4))[0])
        imageSizeZ = int(struct.unpack('i', mrcFile.read(4))[0])
        mrcFile.seek(28)
        mx = int(struct.unpack('i', mrcFile.read(4))[0])
        mrcFile.seek(40)
        xlen = float(struct.unpack('f', mrcFile.read(4))[0])
        apix = xlen / mx
        mrcFile.seek(196)
        originX = float(struct.unpack('f', mrcFile.read(4))[0])
        originY = float(struct.unpack('f', mrcFile.read(4))[0])
        originZ = float(struct.unpack('f', mrcFile.read(4))[0])
    return imageSizeX, imageSizeY, imageSizeZ, apix, originX, originY, originZ

def readMrcData(mrcFileName):
    with open(mrcFileName, "rb") as mrcFile:
        imageSizeX = int(struct.unpack('i', mrcFile.read(4))[0])
        imageSizeY = int(struct.unpack('i', mrcFile.read(4))[0])
        imageSizeZ = int(struct.unpack('i', mrcFile.read(4))[0])
        mrcMode = int(struct.unpack('i', mrcFile.read(4))[0])
        if mrcMode == 2:
            mrcType = np.float32
        elif mrcMode == 12:
            mrcType = np.float16
        mrcData = np.fromfile(mrcFile, dtype=np.dtype(mrcType), count=(imageSizeX * imageSizeY * imageSizeZ),
                              offset=1024 - 16)
    return mrcData

def writeMrcFile(mrcData, stencilFile, outFile):
    with open(stencilFile, "rb") as mrcStencilFile:
        mrcHeader = mrcStencilFile.read(1024)
    with open(outFile, 'wb+') as mrcFile:
        mrcFile.write(mrcHeader)
        mrcFile.seek(12, 0)
        mrcFile.write(b"\x02\x00")
        mrcFile.seek(1024, 0)
        mrcData.astype('float32').tofile(mrcFile)