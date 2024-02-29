function run(){
KEY_NUM=${1:-2}
DATASET=TruthfulQA
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B or Mistral-7B-v0.1
OUTPUT_PATH=/path/to/data//HaluEval/$DATASET/$MODEL

mkdir -p $OUTPUT_PATH
cd ..

python -u tfqa_keys.py \
    --model-name /path/to/model \
    --data-path /path/to/data/$DATASET \
    --output-path $OUTPUT_PATH/keys${KEY_NUM}.json \
    --num-gpus 1 \
    --key-num $KEY_NUM \
| tee -a $OUTPUT_PATH/keys${KEY_NUM}.log

cd experiments
}

run 10
