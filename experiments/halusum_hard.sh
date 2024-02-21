function run(){
ALPHA=${1:-3}
KEY_NUM=${2:-2}
RUN_NAME=alpha${ALPHA}
DATASET=summarization
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B or Mistral-7B-v0.1
OUTPUT_DIR=outputs/HaluEval-$DATASET/$MODEL/keys$KEY_NUM/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp halusum_hard.sh $OUTPUT_DIR/run.sh

cd ..

python -u halusum_eval.py \
    --model-name /path/to/model \
    --data-path /path/to/data/${DATASET}_data.json \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
    --keys-path /path/to/keys \
    --pondering hard \
    --alpha $ALPHA \
| tee -a experiments/$OUTPUT_DIR/run.log

cd experiments
}

run 1.6 4
