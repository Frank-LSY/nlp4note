#!/bin/bash

function read_dir(){
    for file in `ls $1`       #注意此处这是两个反引号，表示运行系统命令
    do
        if [ -d $1"/"$file ]  #注意此处之间一定要加上空格，否则会报错
        then
            mkdir -pv $2"/"$file
            mkdir -pv $3"/"$file
            read_dir $1"/"$file $2"/"$file $3"/"$file
        else

			      a=$[$a+1]
			      b=${1:39}
            echo  -e "\033[43;36m$a/1316484\033[0m" "\033[42;31m$b\033[0m"   #在此处处理文件即可
            len=${#file}
            outfile=${file:0:$[$len-4]}.jsonl

            python3.6 ./bert/extract_features.py \
				--input_file="$1/$file" \
				--output_file="$2/$outfile" \
				--vocab_file=/home/shl183/nlp4note/src/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/vocab.txt \
				--bert_config_file=/home/shl183/nlp4note/src/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/bert_config.json \
				--init_checkpoint=/home/shl183/nlp4note/src/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/model.ckpt-150000.index \
				--layers=-1 \
				--max_seq_length=128 \
				--batch_size=8 \

            fileout=${file:0:$[$len-4]}.pkl
				    python3.6 ./step1.py "$2/$outfile" "$3/$fileout"
				    echo  -e "\033[42;31m"$2/$outfile"\033[0m"
            rm -rf "$2/$outfile"

#			printf "\033c"
        fi
    done
} 


dir_in="/home/shl183/nlp4note/classified_txts/"
dir_out="/home/shl183/nlp4note/bert_embeddings/"
file_out="/home/shl183/nlp4note/file_embeddings/"
read_dir $dir_in $dir_out $file_out




