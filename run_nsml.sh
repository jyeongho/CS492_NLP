#!/usr/bin/env bash

nsml run -m "retro-reader-CA-I-FV-CE" -d korquad-open-ldbd3 -g 2 -c 4 -e run_squad.py -a "--model_type bert --model_name_or_path bert-base-multilingual-cased --do_train --do_eval --fp16 --data_dir train --num_train_epochs 5 --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --output_dir output --overwrite_output_dir --version_2_with_negative"


nsml run -m "retro-reader-E-FV" -d korquad-open-ldbd3 -g 2 -c 4 -e run_cls.py -a "--model_type electra --do_train --do_eval --fp16 --data_dir train --num_train_epochs 5 --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --output_dir output --overwrite_output_dir --version_2_with_negative"


nsml run -m "session for saving models 120, 121" -d korquad-open-ldbd3 -g 1 -c 1 -e run_save_model.py


nsml run -m "ext and diff for 160" -d korquad-open-ldbd3 -g 1 -c 2 -e run_two_model.py -a "--do_eval --fp16 --data_dir train --per_gpu_eval_batch_size 24 --output_dir output --overwrite_output_dir --version_2_with_negative"
