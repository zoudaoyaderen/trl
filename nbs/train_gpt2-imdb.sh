CUDA_VISIBLE_DEVICES=0 nohup python -u run_clm.py \
    --model_name_or_path gpt2 \
    --train_file ../data/imdb-dataset.train.csv \
    --validation_file ../data/imdb-dataset.eval.csv \
    --do_train \
    --do_eval \
    --block_size 256 \
    --output_dir gpt2-imdb > train_gpt2-imdb.log &
