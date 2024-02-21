RUN_NAME=baseline
DATASET=summarization
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B or Mistral-7B-v0.1
OUTPUT_DIR=outputs/HaluEval-$DATASET/$MODEL/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp halusum_baseline.sh $OUTPUT_DIR/run.sh

cd ..

python -u halusum_eval.py \
    --model-name /path/to/model \
    --data-path /path/to/data/${DATASET}_data.json \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
| tee -a experiments/$OUTPUT_DIR/run.log
