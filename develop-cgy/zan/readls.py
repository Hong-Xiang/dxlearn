import struct
with open('test1.ls', 'rb') as f:
    num_lor = struct.unpack('i', f.read(4))
    
    print(num_lor)
