python effmap.py -j master &
sleep 3
CUDA_VISIBLE_DEVICES="0" python tor_rec.py -j worker -t 0 &
CUDA_VISIBLE_DEVICES="1" python tor_rec.py -j worker -t 1 &
#python tor_rec.py -j worker -t 2 &
#python tor_rec.py -j worker -t 3 &
# CUDA_VISIBLE_DEVICES="0" python scanner.py -s 0 -e 12 &
# CUDA_VISIBLE_DEVICES="1" python scanner.py -s 12 -e 24 &
# CUDA_VISIBLE_DEVICES="0" python scanner.py -s 24 -e 36 &
# CUDA_VISIBLE_DEVICES="1" python scanner.py -s 36 -e 50 &
