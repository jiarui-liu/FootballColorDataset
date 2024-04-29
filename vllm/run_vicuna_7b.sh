# vicuna 7b v1.5
# CUDA_VISIBLE_DEVICES=2 bash run_vicuna_7b.sh > logs/stdout_vicuna_7b.txt 2> logs/stderr_vicuna_7b.txt
MODEL_DIR="lmsys/vicuna-7b-v1.5"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=9797 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/vicuna.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=4096