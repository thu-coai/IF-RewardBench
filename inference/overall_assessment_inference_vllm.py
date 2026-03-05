from vllm import LLM, SamplingParams
import argparse
import json
import re
import os
from transformers import AutoTokenizer
from prompts.overall_assessment import critique_generation_prompt


def build_prompt(item, response_a, response_b):
    messages = item["messages"]
    system_prompt = ""
    history = ""
    cnt = 1
    for turn in messages[:-1]:
        if turn["role"] == "system":
            system_prompt = turn["content"]
        elif turn["role"] == "user":
            if history != "":
                history += "\n\n"
            history += f"[第{cnt}轮用户指令-开始]\n{turn['content'].strip()}\n[第{cnt}轮用户指令-结束]"
        elif turn["role"] == "assistant":
            if history != "":
                history += "\n\n"
            history += f"[第{cnt}轮人工智能助手的回复-开始]\n{turn['content'].strip()}\n[第{cnt}轮人工智能助手的回复-结束]"
            cnt += 1
    user_prompt = messages[-1]["content"]
    return critique_generation_prompt.format(system_prompt=system_prompt, history=history, user_prompt=user_prompt, response_a=response_a, response_b=response_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/Qwen/QwQ-32B")
    parser.add_argument("--model_name", type=str, default="QwQ-32B")
    parser.add_argument("--input_path", type=str, default="../data/if_rewardbench.json")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, dtype='bfloat16', trust_remote_code=True)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 30,
        "max_tokens": 16384,
        "skip_special_tokens": True,
        "logprobs" : True,
    }
    sampling_params = SamplingParams(**params_dict)
    
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    with open("position_maps_examples.json", "r", encoding="utf-8") as f:
        position_maps = json.load(f)
    
    outs = []
    prompts = []
    for i, d in enumerate(data):
        for u in range(len(d["responses"])):
            for v in range(len(d["responses"])):
                if u == v:
                    continue
                position = position_maps[str(d["id"])][f"{min(u, v)}_{max(u, v)}"]
                if (position == 0 and u < v) or (position == 1 and u > v):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": build_prompt(d, d["responses"][u]["response"], d["responses"][v]["response"])}
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    cnt = 0
    for i, d in enumerate(data):
        data[i]["pairwise_evaluation_results"] = data[i].get("pairwise_evaluation_results", {})
        for u in range(len(d["responses"])):
            for v in range(len(d["responses"])):
                if u == v:
                    continue
                position = position_maps[str(d["id"])][f"{min(u, v)}_{max(u, v)}"]
                if (position == 0 and u < v) or (position == 1 and u > v):
                    data[i]["pairwise_evaluation_results"][f"{u}_{v}"] = outputs[cnt].outputs[0].text
                    cnt += 1

    with open(f"overall_assessment_results/{args.model_name}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        