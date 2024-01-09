# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from pathlib import Path
import time

DEVICE_ID = 0
DEVICE = "cuda:%d" % DEVICE_ID if torch.cuda.is_available() else "cpu"

import GPUtil


def get_gpu_used_memory(device_id):
    byte_gpu_used_memory = GPUtil.getGPUs()[device_id].memoryUsed
    GB_gpu_used_memory = byte_gpu_used_memory / 1024
    return GB_gpu_used_memory


from torchinfo import summary


def get_billion_param(model):
    billion = 1000**3
    return round(summary(model, verbose=0).total_params / billion, 2)


def run_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="D:/models",
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    text = "Hello, world!"
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

    if torch.cuda.is_available():
        token_ids = token_ids.to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            token_ids,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(output_ids.tolist()[0])

    return output, model, tokenizer


def get_gpu_memory_usage_and_num_param(model_name):
    pre_gpu_usage = get_gpu_used_memory(DEVICE_ID)

    predictions, model, tokenizer = run_model(model_name)
    print("predictions: ", predictions)

    post_gpu_mem_usage = get_gpu_used_memory(DEVICE_ID)
    used_gpu_memory = round(post_gpu_mem_usage - pre_gpu_usage, 2)

    num_param = get_billion_param(model)

    # gpu memory release
    del predictions, model, tokenizer
    torch.cuda.empty_cache()

    return used_gpu_memory, num_param


def main():
    f_name = "data.pkl"

    if not Path(f_name).exists():
        data = {"model name": [], "num_param(Billion)": [], "gpu_memory_usage": []}
        df = pd.DataFrame(data)
        df.set_index("model name", inplace=True)
        df.to_pickle(f_name)

    df = pd.read_pickle(f_name)

    # 正確に検証するには、メモリをきちんと開放させるため
    # １モデルずつ実行したほうが良い。
    model_names = [
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "microsoft/phi-2",
        # "llama-moe/LLaMA-MoE-v1-3_5B-4_16",
        # "rinna/youri-7b",
        # "mistralai/Mistral-7B-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.1",
        # "HuggingFaceH4/zephyr-7b-beta",
        # "upstage/SOLAR-10.7B-Instruct-v1.0",
        # "facebook/xglm-4.5B",
        # "01-ai/Yi-6B",
        "Qwen/Qwen-1_8B-Chat",
    ]

    for model_name in model_names:
        gpu_usage, num_param = get_gpu_memory_usage_and_num_param(model_name)
        print(model_name, num_param, gpu_usage)
        df.loc[model_name] = [num_param, gpu_usage]

    df.to_pickle(f_name)

main()
# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle("data.pkl")
df.sort_values(by="num_param(Billion)", inplace=True)
print(df)
plt.figure(figsize=(10, 10))
plt.xlabel("num param(Billion)")
plt.ylabel("GPU memory usage(GB)")
plt.plot(df["num_param(Billion)"], df["gpu_memory_usage"], "o-")
plt.savefig("result.png")