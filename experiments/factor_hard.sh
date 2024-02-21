function run(){
ALPHA=${1:-3}
KEY_NUM=${2:-2}
RUN_NAME=alpha${ALPHA}
DATASET=news_factor
MODEL=LLaMA_hf_7B
OUTPUT_DIR=outputs/$DATASET/$MODEL/keys${KEY_NUM}/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp factor_hard.sh $OUTPUT_DIR/run.sh

cd ..

CUDA_VISIBLE_DEVICES=7 python -u factor_eval.py \
    --model-name /path/to/model \
    --data-path /path/to/data/$DATASET.csv \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
    --keys-path /path/to/keys \
    --pondering hard \
    --alpha $ALPHA \
| tee -a experiments/$OUTPUT_DIR/run.log

cd experiments
}

run 0.1 8
