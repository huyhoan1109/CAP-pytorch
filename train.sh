pip install -r requirements.txt
python3 download.py --dataset voc2012
python3 gen_meta.py --dataset voc2012 --labeled 0.15 --valid 0.15 --test 0.15
python3 run.py --dataset voc2012 --train t --use-ema t --resume f 