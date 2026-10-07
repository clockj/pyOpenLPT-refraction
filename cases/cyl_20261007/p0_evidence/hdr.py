import struct, sys, os
def header(path):
    b = open(path, "rb").read()
    i = b.find(struct.pack("<HH", 1280, 800))
    serial, = struct.unpack_from("<I", b, i + 6)
    fps, shutter = struct.unpack_from("<II", b, i + 31)
    return len(b), serial, fps, shutter
for p in sys.argv[1:]:
    print(p, header(p))
