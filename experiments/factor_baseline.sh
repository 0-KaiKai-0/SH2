RUN_NAME=baseline
DATASET=news_factor # or wiki_factor
MODEL=LLaMA_hf_7B # or LLaMA2_hf_7B or Mistral-7B-v0.1
OUTPUT_DIR=outputs/$DATASET/$MODEL/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp factor_baseline.sh $OUTPUT_DIR/run.sh

cd ..

CUDA_VISIBLE_DEVICES=0 python -u factor_eval.py \
    --model-name /path/to/model \
    --data-path /path/to/data/$DATASET.csv \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
| tee -a experiments/$OUTPUT_DIR/run.log
