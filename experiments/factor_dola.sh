RUN_NAME=dola
DATASET=news_factor
MODEL=LLaMA_hf_7B
OUTPUT_DIR=outputs/$DATASET/$MODEL/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp factor_dola.sh $OUTPUT_DIR/run.sh

cd ..

CUDA_VISIBLE_DEVICES=1 python -u factor_eval.py \
    --model-name /home/jskai/workspace/models/$MODEL \
    --data-path /home/jskai/workspace/factor/data/$DATASET.csv \
    --early-exit-layers 0,2,4,6,8,10,12,14,32 \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
| tee -a experiments/$OUTPUT_DIR/run.log
