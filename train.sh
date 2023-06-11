python3 download.py --dataset voc2012
python3 gen_meta.py --dataset voc2012
python3 run.py --dataset voc2012 --train t --use-ema t --resume f --use-asl t --opt 3 --sch 3