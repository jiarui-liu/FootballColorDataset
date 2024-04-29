# bash run_llama2_70b_tgi.sh > logs/stdout_llama2_70b_tgi.txt 2> logs/stderr_llama2_70b_tgi.txt

MODEL_DIR="meta-llama/Llama-2-70b-chat-hf/"

text-generation-launcher --model-id $MODEL_DIR --port 9570 --max-input-length 3072 --max-total-tokens 4096