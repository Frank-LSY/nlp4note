#!/bin/bash

function read_dir(){
    for file in `ls $1`       #注意此处这是两个反引号，表示运行系统命令
    do
        if [ -d $1"/"$file ]  #注意此处之间一定要加上空格，否则会报错
        then
        	mkdir -pv $2"/"$file
            read_dir $1"/"$file $2"/"$file
        else

			a=$[$a+1]
			b=${1:29}
            echo  -e "\033[43;36m$a/1316484\033[0m" "\033[42;31m$b\033[0m"   #在此处处理文件即可
            outfile=${file:0:8}.jsonl

            python3.6 ../bert/extract_features.py \
				--input_file="$1/$file" \
				--output_file="$2/$outfile" \
				--vocab_file=/home/shl183/nlp4note/src/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/vocab.txt \
				--bert_config_file=/home/shl183/nlp4note/src/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/bert_config.json \
				--init_checkpoint=/home/shl183/nlp4note/src/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/model.ckpt-150000.index \
				--layers=-1 \
				--max_seq_length=128 \
				--batch_size=8 \
				> "/dev/null"
			printf "\033c"
        fi
    done
} 


dir_in="/home/shl183/nlp4note/classified_txt/NOTES-ER_REPORT"
dir_out="/home/shl183/nlp4note/bert_embedding/NOTES-ER_REPORT"
read_dir $dir_in $dir_out




