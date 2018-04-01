
#python tor_rec.py -j worker -t 2 &
#python tor_rec.py -j worker -t 3 &
CUDA_VISIBLE_DEVICES="0" python scanner.py -s 0 -e 52 &
CUDA_VISIBLE_DEVICES="1" python scanner.py -s 135 -e  &
CUDA_VISIBLE_DEVICES="0" python scanner.py -s 270 -e 312 &
CUDA_VISIBLE_DEVICES="1" python scanner.py -s 405 -e 416 &
