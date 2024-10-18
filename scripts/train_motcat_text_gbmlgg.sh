CUDA_VISIBLE_DEVICES=3 python main.py \
--data_root_dir data/GBMLGG/patch_feats/clam_inr50t_s20 \
--split_dir tcga_gbmlgg \
--mode coattn_text \
--model_type motcat_text \
--bs_micro 16385 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig \
--status_path data/GBMLGG/status_gbmlgg.json