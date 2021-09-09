#!/bin/bash

# DF-LSL
python3 fewshot/new_run_cl.py --n 1 --log_dir exp/acl/df_lsl --lsl --glove_init --lang_lambda 5 --max_lang_per_class 20 --sample_class_lang --n_shot 4
