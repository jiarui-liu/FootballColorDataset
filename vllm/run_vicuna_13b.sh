# vicuna 13b v1.5
# CUDA_VISIBLE_DEVICES=3 bash run_vicuna_13b.sh > logs/stdout_vicuna_13b.txt 2> logs/stderr_vicuna_13b.txt
MODEL_DIR="lmsys/vicuna-13b-v1.5"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=9798 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/vicuna.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=4096