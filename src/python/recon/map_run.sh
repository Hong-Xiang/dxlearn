
#python tor_rec.py -j worker -t 2 &
#python tor_rec.py -j worker -t 3 &
CUDA_VISIBLE_DEVICES="0" python scanner.py -s 0 -e 135 &
CUDA_VISIBLE_DEVICES="1" python scanner.py -s 135 -e 270 &
CUDA_VISIBLE_DEVICES="0" python scanner.py -s 270 -e 405 &
CUDA_VISIBLE_DEVICES="1" python scanner.py -s 405 -e 540 &
