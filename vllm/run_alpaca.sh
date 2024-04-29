# alpaca
# CUDA_VISIBLE_DEVICES=3 bash run_alpaca.sh > logs/stdout_alpaca.txt 2> logs/stderr_alpaca.txt
MODEL_DIR="chavinlo/alpaca-native"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=6767 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/alpaca.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=4096