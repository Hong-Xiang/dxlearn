# python effmap.py -j master &
# sleep 3
# CUDA_VISIBLE_DEVICES="0" python effmap.py -j worker -t 0 &
# CUDA_VISIBLE_DEVICES="1" python effmap.py -j worker -t 1 &
#python tor_rec.py -j worker -t 2 &
#python tor_rec.py -j worker -t 3 &
CUDA_VISIBLE_DEVICES="0" python scanner.py -s 0 -e 25 &
# CUDA_VISIBLE_DEVICES="1" python scanner.py -s 108 -e 216 &
# CUDA_VISIBLE_DEVICES="0" python scanner.py -s 216 -e 324 &
# CUDA_VISIBLE_DEVICES="1" python scanner.py -s 324 -e 432 &
