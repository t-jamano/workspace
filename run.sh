# 1M_EN_QQ_log 7000 for 128 batch_size
# QD has unlimiated data
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=0 python experiments.py --model kate2_qd3_dssm --dataset 30M_QD_lower2.txt --b 32 --e 25 --a 1 --o adadelta
# CUDA_VISIBLE_DEVICES=0 python experiments.py --model kate2_qd3_dssm --dataset 30M_EN_pos_qd_log --b 32 --e 25 --o adadelta
# CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=0 python experiments.py --model kate2_bpe --dataset 30M_QD_lower2.txt --b 32 --e 25 --a 1 --o adadelta

# python experiments.py --model vae_dssm --dataset 30M_EN_pos_qd_log --i 40000 --b 256 --e 10


