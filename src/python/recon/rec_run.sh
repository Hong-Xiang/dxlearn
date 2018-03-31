python tor_rec.py -j master &
sleep 3
CUDA_VISIBLE_DEVICES="0" python tor_rec.py -j worker -t 0 &
CUDA_VISIBLE_DEVICES="1" python tor_rec.py -j worker -t 1 &