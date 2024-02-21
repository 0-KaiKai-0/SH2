function run(){
KEY_NUM=${1:-2}
DATASET=summarization
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B or Mistral-7B-v0.1
OUTPUT_PATH=/home/jskai/workspace/datasets/HaluEval/$DATASET/$MODEL

mkdir -p $OUTPUT_PATH
cd ..

python -u halusum_keys.py \
    --model-name /path/to/model \
    --data-path /path/to/data/${DATASET}_data.json \
    --output-path $OUTPUT_PATH/keys${KEY_NUM}.json \
    --num-gpus 1 \
    --key-num $KEY_NUM \
| tee -a $OUTPUT_PATH/keys$KEY_NUM.log

cd experiments
}

run 4
