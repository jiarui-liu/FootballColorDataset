# llama2 13b chat hf
# CUDA_VISIBLE_DEVICES=1 bash run_llama2_13b.sh > logs/stdout_llama2_13b.txt 2> logs/stderr_llama2_13b.txt
MODEL_DIR="meta-llama/Llama-2-13b-chat-hf"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=2526 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/llama2.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=4096