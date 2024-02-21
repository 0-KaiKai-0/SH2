SH2: Self-Highlighted Hesitation Helps You Decode More Truthfully
===

## Environment setup

```
pip install -e transformers
pip install datasets
pip install accelerate
pip install openai # -> only for truthfulqa and gpt4_eval
```

## Example experiments
```
cd experiments # running scripts
```

- Baseline: LLaMA_hf_7B, LLaMA2_hf_7B or Mistral-7B-v0.1
```
bash factor_baseline.sh
```

- DoLa
```
bash factor_dola.sh
```

- SH2
```
bash factor_keys.sh # prepare key tokens
bash factor_hard.sh
```
