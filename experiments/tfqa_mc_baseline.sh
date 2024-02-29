RUN_NAME=baseline
DATASET=TruthfulQA
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B or Mistral-7B-v0.1
OUTPUT_DIR=outputs/$DATASET/$MODEL/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp tfqa_mc_baseline.sh $OUTPUT_DIR/run.sh

cd ..

python -u tfqa_mc_eval.py \
    --model-name /path/to/model \
    --data-path /path/to/data/$DATASET \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
| tee -a experiments/$OUTPUT_DIR/run.log
