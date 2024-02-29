function run(){
KEY_NUM=${1:-2}
DATASET=news_factor
MODEL=LLaMA_hf_7B
OUTPUT_PATH=/path/to/data/$DATASET/$MODEL

mkdir -p $OUTPUT_PATH
cd ..

python -u factor_keys.py \
    --model-name /path/to/model \
    --data-path /path/to/data/$DATASET.csv \
    --output-path $OUTPUT_PATH/keys${KEY_NUM}.json \
    --num-gpus 1 \
    --key-num $KEY_NUM \
| tee -a $OUTPUT_PATH/keys$KEY_NUM.log

cd experiments
}

run 8
