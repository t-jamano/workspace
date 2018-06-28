# 1M_EN_QQ_log 7000 for 128 batch_size
# QD has unlimiated data
python experiments.py --model dssm --dataset 1M_EN_QQ_log --i 3500 --b 256 --e 10
# python experiments.py --model vae_dssm --dataset 30M_EN_pos_qd_log --i 40000 --b 256 --e 10