from vllm import LLM, SamplingParams
import argparse
import json
import re
import os
from transformers import AutoTokenizer
from prompts.constraint_assessment import critique_generation_prompt

def parse_solution(text):
    return text.split("</think>")[-1].strip()


def reconstruct(checklist):    
    prompt = ""
    if checklist == None:
        return prompt
    for i, c in enumerate(checklist):
        if prompt != "":
            prompt += "\n\n"
        prompt += f"[检查项{i+1}-开始]\n{c.strip()}\n[检查项{i+1}-结束]"
    return prompt.strip()


def build_prompt(item, response):
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
    return critique_generation_prompt.format(system_prompt=system_prompt, history=history, user_prompt=user_prompt, assistant_response=response, checklist=reconstruct(item["checklist"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/Qwen/QwQ-32B")
    parser.add_argument("--model_name", type=str, default="QwQ-32B")
    parser.add_argument("--input_path", type=str, default="../data/if_rewardbench_examples.json")
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
    outs = []
    prompts = []
    for i, d in enumerate(data):
        for j, r in enumerate(d["responses"]):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": build_prompt(d, r["response"])}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
    outputs = llm.generate(prompts, sampling_params)

    cnt = 0
    outs = []
    for i, d in enumerate(data):
        for j, r in enumerate(d["responses"]):
            generated_text = outputs[cnt].outputs[0].text
            data[i]["responses"][j]["critique"] = generated_text
            cnt += 1

    with open(f"constraint_assessment_results/{args.model_name}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        