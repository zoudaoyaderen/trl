
<<COMMENT
COMMENT

data=nlg_data.data
awk -F'\t' 'BEGIN{srand()}{print rand()"\t"$0}' $data | sort -k1g | cut -f2- > tmp
mv tmp nlg_data.data.rdm

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/main.py train > model.0/1.log &
