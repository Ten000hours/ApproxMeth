# in baselines folder
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_teacher.py --task_name STSB --metric_name eval_combined_score --model_name bert-base-uncased

# --do_train for approxied model
CUDA_VISIBLE_DEVICES=4,5,6,7 python baseline.py --task_name STSB --model_path ~/yuanzhong/MPCFormer/src/baselines/tmp/exp/STSB/bert-base-uncased/5e-05 --baseline_type S1 --hidden_act smu1_train --softmax_act softmax

# in main folder
# distill with approxied model
CUDA_VISIBLE_DEVICES=4,5,6,7 python exp2.py --task_name STSB --teacher_dir ~/yuanzhong/MPCFormer/src/baselines/tmp/exp/STSB/bert-base-uncased/5e-05 --student_dir  ~/yuanzhong/MPCFormer/src/baselines/tmp/baseline/STSB/bert-base-uncased/smu1_train/softmax/HPO_S1/5e-06/256/100 --hidden_act smu1_train --softmax_act softmax

# in baselines folder
# --do_eval for accuracy check 
# check plaintext model
CUDA_VISIBLE_DEVICES=4,5,6,7 python baseline.py --task_name STSB --model_path ~/yuanzhong/MPCFormer/src/main/tmp/distill/STSB/smu1_train_softmax/bert-base-uncased/5e-05_1e-05_32_stage2 --baseline_type S1 --hidden_act smu1_train --softmax_act softmax

# check our approximate model
CUDA_VISIBLE_DEVICES=4,5,6,7 python baseline.py --task_name MRPC --model_path ~/yuanzhong/MPCFormer/src/main/tmp/distill/MRPC/smu1_train_softmax/bert-base-uncased/5e-05_1e-05_32_stage2 --baseline_type S1 --hidden_act smu1_infer --softmax_act softmax