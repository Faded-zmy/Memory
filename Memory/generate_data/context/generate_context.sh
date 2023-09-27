screen -S context2
conda activate zmy
CUDA_VISIBLE_DEVICES=3 python3 generate_context_from_60k_complete_questions.py --split_id 2