RUN_NAME=baseline
DATASET=news_factor
MODEL=LLaMA_hf_7B
OUTPUT_DIR=outputs/$DATASET/$MODEL/$RUN_NAME-pdb

mkdir -p $OUTPUT_DIR
cp factor_baseline.sh $OUTPUT_DIR/run.sh

cd ..

CUDA_VISIBLE_DEVICES=0 python -u factor_eval.py \
    --model-name /home/jskai/workspace/models/$MODEL \
    --data-path /home/jskai/workspace/factor/data/$DATASET.csv \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
| tee -a experiments/$OUTPUT_DIR/run.log
