torchrun \
  --nproc_per_node 2 \
  --master_port 29410 \
  train_llada_sft.py \
  --model_name "GSAI-ML/LLaDA-8B-Instruct" \
  --output_dir logs/tmp_logs_llada_sft_$(date +%Y%m%d_%H%M%S) \
  --num_epochs 2 \
  --debugging
