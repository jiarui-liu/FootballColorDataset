import transformers
import torch

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
# model_name = "lmsys/vicuna-13b-v1.5"
# model_name = "chavinlo/alpaca-native"

cache_dir = "<cache_dir>"
access_token = "<access_token>"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    # token=access_token,
    cache_dir=cache_dir
)

model = transformers.AutoModel.from_pretrained(
    model_name,
    # token=access_token,
    trust_remote_code=True,
    cache_dir=cache_dir
)