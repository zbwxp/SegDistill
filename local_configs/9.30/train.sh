bash tools/dist_train.sh local_configs/9.30/attention_mse.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/attention_s.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/attention_c.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/feature_mse.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/feature_s.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/feature_c.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/logits_c_mask.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/logits_cg3+sg_k5_s5.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/logits_sg_k3_s3.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.30/logits_sg_k5_s5.py 8;
sleep 10;