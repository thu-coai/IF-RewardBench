import argparse
import json
import os
import random
import numpy as np
import math
from collections import defaultdict

small_oss = ["MiniCPM4.1-8B", "Qwen3-8B", "GLM-4-9B-0414", "Llama-3.1-Tulu-3.1-8B"]
middle_oss = ["GLM-4.5-Air", "LLama-3.3-70B-Instruct", "GPT-OSS-20B", "Qwen3-32B"]
large_oss = ["Qwen-3-235B-A22B-Instruct-2507", "Kimi-K2", "GLM-4.5", "DeepSeek-R1"]
api_models = ["Gemini-2.5-Pro", "Doubao-1.6-Seed", "GPT-5", "Claude-4.5-Sonnet"]


prompt_composition_type_maps = {}
position_maps = {} 

def compute_elo_with_indices(n, comparisons, k_factor=32, initial_rating=1200, epochs=100):
    ratings = [float(initial_rating)] * n
    data = list(comparisons)
    
    for epoch in range(epochs):
        random.shuffle(data)
        current_k = k_factor
        if epoch > epochs * 0.2:
            decay = (epoch - epochs * 0.2) / (epochs * 0.8)
            current_k = max(1.0, k_factor * (1 - decay))

        for x, y in data:
            loser_id = x
            winner_id = y
            r_loser = ratings[loser_id]
            r_winner = ratings[winner_id]
            
            expected_winner = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))
            expected_loser = 1 / (1 + 10 ** ((r_winner - r_loser) / 400))
            
            ratings[winner_id] = r_winner + current_k * (1 - expected_winner)
            ratings[loser_id] = r_loser + current_k * (0 - expected_loser)
            
    return ratings


def get_model_type(model):
    if model in small_oss: return "Small"
    elif model in middle_oss: return "Middle"
    elif model in large_oss: return "Large"
    elif model in api_models: return "API"
    else: return "Unknown"


def get_constraint_count_bucket(count):
    if count <= 3: return "<=3"
    if count == 4: return "4"
    if count == 5: return "5"
    if count == 6: return "6"
    return ">=7"


def get_turn(item):
    messages = item["messages"]
    if messages[0]["role"] == "system":
        messages = messages[1:]
    return min((len(messages) + 1) // 2, 5)


def get_pairwise_result(item, u, v):
    has_A = "[[A]]" in item["pairwise_evaluation_results"][f"{u}_{v}"]
    has_B = "[[B]]" in item["pairwise_evaluation_results"][f"{u}_{v}"]
    if has_A and not has_B: return "A"
    if has_B and not has_A: return "B"
    return "C"


def calculate_metric_stats(predictions):
    acc_pair = 0
    all_pair = 0
    C, D, T_y = 0, 0, 0
    for pred in predictions:
        all_pair += 1
        if pred == "A":
            acc_pair += 1
            C += 1
        else:
            if pred == "B":
                D += 1
            else:
                T_y += 1
    denominator_kendall = max(1, math.sqrt((C + D) * (C + D + T_y)))
    return {
        "Accuracy": acc_pair / max(all_pair, 1),
        "Kendall Taub": (C - D) / denominator_kendall,
    }


def get_pref_from_scores(scores, a_id, b_id):
    if scores[a_id] > scores[b_id]:
        return "A"
    elif scores[b_id] > scores[a_id]:
        return "B"
    else:
        return "C"


def calculate_pairwise_metrics(item):
    global position_maps
    
    comparisons = []
    for u in range(len(item["responses"])):
        for v in range(len(item["responses"])):
            if u == v: continue
            position = position_maps[str(item["id"])][f"{min(u, v)}_{max(u, v)}"]
            if (position == 0 and (u < v)) or (position == 1 and (u > v)):
                preference = get_pairwise_result(item, u, v)
                if preference == "C":
                    continue
                if preference == "A":
                    comparisons.append((v, u))
                else:
                    comparisons.append((u, v))
            
    scores = compute_elo_with_indices(len(item["responses"]), comparisons)
    preds = []
    for p in item["preference_graph"]:
        preds.append(get_pref_from_scores(scores, p["chosen"]["response_id"], p["rejected"]["response_id"]))
    return calculate_metric_stats(preds)


class StatsCollector:
    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def update(self, group_name, key, pairwise_res):
        target = self.stats[group_name][key]
        target['pair_acc'].append(pairwise_res["Accuracy"])
        target['kendall'].append(pairwise_res["Kendall Taub"])

    def compute_metrics(self, group_name):
        output = {}
        keys = list(self.stats[group_name].keys())
        
        def safe_sort_key(k):
            try:
                if isinstance(k, str):
                    clean_k = k.replace('.', '', 1)
                    if clean_k.isdigit():
                        return (0, float(k))
                elif isinstance(k, (int, float)):
                    return (0, float(k))
            except:
                pass
            return (1, str(k))

        keys.sort(key=safe_sort_key)

        for key in keys:
            data = self.stats[group_name][key]
            output[key] = {
                "ranking": {
                    "Accuracy": np.mean(data['pair_acc']) if data['pair_acc'] else 0,
                    "Kendall Taub": np.mean(data['kendall']) if data['kendall'] else 0,
                }
            }
        return output
    
    def get_raw_means(self, group_name, key, metric_type):
        if key not in self.stats[group_name]: return 0
        mapping = {
            'pair_acc': 'pair_acc', 
            'kendall': 'kendall'
        }
        vals = self.stats[group_name][key][mapping[metric_type]]
        return np.mean(vals) if vals else 0
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position_maps_path", type=str, default="../inference/position_maps.json")
    parser.add_argument("--evaluation_results_path", type=str, default="../inference/overall_assessment_results")
    args = parser.parse_args()
    with open(args.position_maps_path, "r", encoding="utf-8") as f:
        position_maps = json.load(f)
    
    random.seed(42)
    files = [f for f in os.listdir(args.evaluation_results_path) if f.endswith(".json")]
    for file in files:
        print(f"Processing {file}...")
        with open(f"{args.evaluation_results_path}/{file}", "r", encoding="utf-8") as f:
            data = json.load(f)
        collector = StatsCollector()
        for i, d in enumerate(data):
            c_bucket = get_constraint_count_bucket(len(d["checklist"]))
            all_metrics_res = calculate_pairwise_metrics(d)
            
            groupings = [
                ("model_type", get_model_type(d["response_generation_model"])),
                ("instruction_type", d["instruction_type"]),
                ("turn", get_turn(d)),
                ("constraint_count", c_bucket), 
            ]

            for group, key in groupings:
                collector.update(group, key, all_metrics_res)

        metrics = {}
        metrics["metrics_per_model_type"] = collector.compute_metrics("model_type")
        metrics["metrics_per_turn"] = collector.compute_metrics("turn")
        metrics["metrics_per_instruction_type"] = collector.compute_metrics("instruction_type")
        metrics["metrics_per_constraint_count"] = collector.compute_metrics("constraint_count")
        metrics["Overall"] = {}
        metrics["Overall"]["ranking"] = {}
    
        for k_out, k_raw in [("Accuracy", "pair_acc"), ("Kendall Taub", "kendall")]:
            valid_scores = []
            for t_type in ["Single_Turn", "Multi_Turn", "System_Prompt"]:
                if t_type in collector.stats["instruction_type"]:
                    val = collector.get_raw_means("instruction_type", t_type, k_raw)
                    valid_scores.append(val)
            metrics["Overall"]["ranking"][k_out] = np.mean(valid_scores) if valid_scores else 0

        with open(f"overall_assessment_results/{file}", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
    