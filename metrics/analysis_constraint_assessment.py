import re
import argparse
import json
import os
import numpy as np
import math
from sklearn.metrics import matthews_corrcoef
from collections import defaultdict

constraint_composition_type_maps = {}
prompt_composition_type_maps = {}

def parse_critique(critique):
    if critique == None:
        return []
    pattern = re.compile(
        r'\[检查项(?P<编号>\d+)-开始\]\s*'
        r'\n要求：(.*?)\s*'
        r'\n分析：(.*?)\s*'
        r'\n结论：(.*?)\s*'
        r'\n\[检查项\1-结束\]',
        re.S
    )
    matches = pattern.finditer(critique)
    results = []
    for match in matches:
        results.append({
            '编号': f"{match.group('编号')}",
            '要求': match.group(2).strip(),
            '分析': match.group(3).strip(),
            '结论': match.group(4).strip()
        })
    if critique.count('-开始]') != len(results) or critique.count('-结束]') != len(results):
        return []
        
    return results


def verdict(critique):
    critique = parse_solution(critique)
    if any(x in critique for x in ["[[人工智能助手的回复满足了该要求]]", "[[满足该要求]]", "人工智能助手的回复满足了该要求"]):
        return 1
    elif "[[人工智能助手的回复没有满足该要求]]" in critique:
        return 0
    return 0


def get_label(responses, length):
    labels = []
    critique = parse_critique(responses["critique"])
    for c in critique:
        labels.append(verdict(c["结论"]))
    if len(labels) < length:
        labels += [1] * (length - len(labels))
    return labels[:length]


small_oss = ["MiniCPM4.1-8B", "Qwen3-8B", "GLM-4-9B-0414", "Llama-3.1-Tulu-3.1-8B"]
middle_oss = ["GLM-4.5-Air", "LLama-3.3-70B-Instruct", "GPT-OSS-20B", "Qwen3-32B"]
large_oss = ["Qwen-3-235B-A22B-Instruct-2507", "Kimi-K2", "GLM-4.5", "DeepSeek-R1"]
api_models = ["Gemini-2.5-Pro", "Doubao-1.6-Seed", "GPT-5", "Claude-4.5-Sonnet"]


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


def parse_solution(text):
    try:
        return text.split("</think>")[-1].strip()
    except:
        return text


def calculate_pointwise_metrics(hyp, ref):
    if len(hyp) != len(ref):
        raise ValueError("The lengths of hyp and ref must be the same.")
    if len(hyp) == 0:
        return {"Positive F1": 0, "Negative F1": 0}
    
    TP = sum(1 for h, r in zip(hyp, ref) if h == 1 and r == 1)
    TN = sum(1 for h, r in zip(hyp, ref) if h == 0 and r == 0)
    FP = sum(1 for h, r in zip(hyp, ref) if h == 1 and r == 0)
    FN = sum(1 for h, r in zip(hyp, ref) if h == 0 and r == 1)
    
    prec_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_pos = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_pos = (2 * prec_pos * rec_pos) / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0
    
    prec_neg = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec_neg = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_neg = (2 * prec_neg * rec_neg) / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0
    
    return {"Positive F1": f1_pos, "Negative F1": f1_neg}


def calculate_pairwise_metrics(pairs, scores):
    edges = set()
    nodes = set()
    for p in pairs:
        ra, rb = p["chosen"]["response_id"], p["rejected"]["response_id"]
        edges.add((ra, rb))
        nodes.add(ra)
        nodes.add(rb)
    
    acc_pair = 0
    all_pair = 0
    golden_labels = []
    pred_labels = []
    C, D, T_y = 0, 0, 0
    for e in edges:
        u, v = e[0], e[1]
        golden_labels.extend([1, -1]) 
        if scores[u] > scores[v]:
            acc_pair += 1
            pred_labels.extend([1, -1])
            C += 1
        elif scores[u] == scores[v]:
            pred_labels.extend([0, 0])
            T_y += 1
        else:
            pred_labels.extend([-1, 1])
            D += 1
        all_pair += 1
            
    denominator_kendall = max(1, math.sqrt((C + D) * (C + D + T_y)))
    return {
        "Accuracy": acc_pair / max(all_pair, 1),
        "Kendall Taub": (C - D) / denominator_kendall,
    }



def calculate_mcc(refs, hyps):
    return matthews_corrcoef(refs, hyps) if len(refs) > 0 else 0


class StatsCollector:
    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def update(self, group_name, key, pointwise_res, pairwise_res, golden, pred):
        target = self.stats[group_name][key]
        target['p_f1_macro'].append(pointwise_res["Positive F1"])
        target['n_f1_macro'].append(pointwise_res["Negative F1"])
        target['golden'].extend(golden)
        target['pred'].extend(pred)
        target['pair_acc'].append(pairwise_res["Accuracy"])
        target['kendall'].append(pairwise_res["Kendall Taub"])

    def compute_metrics(self, group_name):
        output = {}
        keys = list(self.stats[group_name].keys())
        try:
            keys.sort(key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
        except:
            keys.sort()

        for key in keys:
            data = self.stats[group_name][key]
            if group_name == "source" and key == "Overall": continue
            
            output[key] = {
                "verification": {
                    "Positive F1": np.mean(data['p_f1_macro']) if data['p_f1_macro'] else 0,
                    "Negative F1": np.mean(data['n_f1_macro']) if data['n_f1_macro'] else 0,
                },
                "ranking": {
                    "Accuracy": np.mean(data['pair_acc']) if data['pair_acc'] else 0,
                    "Kendall Taub": np.mean(data['kendall']) if data['kendall'] else 0,
                }
            }
        return output

    def get_raw_means(self, group_name, key, metric_type):
        if key not in self.stats[group_name]: return 0
        mapping = {
            'pf1': 'p_f1_macro', 
            'nf1': 'n_f1_macro',
            'pair_acc': 'pair_acc', 
            'kendall': 'kendall'
        }
        vals = self.stats[group_name][key][mapping[metric_type]]
        return np.mean(vals) if vals else 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/if_rewardbench.json")
    parser.add_argument("--evaluation_results_path", type=str, default="../inference/constraint_assessment_results")
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    constraint_composition_type_maps = {}
    constraint_category_maps = {}
    for d in data:
        for c in d["constraint_type"]:
            constraint_composition_type_maps[c["item"].strip()] = c["constraint_composition_types"]
            constraint_category_maps[c["item"].strip()] = c["constraint_categories"]

    files = [f for f in os.listdir(args.evaluation_results_path) if f.endswith(".json")]
    for file in files:
        print(f"Processing {file}...")
        with open(f"{args.evaluation_results_path}/{file}", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        collector = StatsCollector()
        cat_golden = defaultdict(list)
        cat_pred = defaultdict(list)
        comp_golden = defaultdict(list)
        comp_pred = defaultdict(list)

        for d in data:
            c_bucket = get_constraint_count_bucket(len(d["checklist"]))
            scores = {}
            golden_labels = []
            pred_labels = []
            for i, resp in enumerate(d["responses"]):
                pl = get_label(resp, len(resp["labels"]))
                gl = resp["labels"]
                scores[resp["response_id"]] = np.mean(pl)
                golden_labels.extend(gl)
                pred_labels.extend(pl)
                
                for k, constraint in enumerate(d["checklist"]):
                    cats = constraint_category_maps[constraint.strip()]
                    comp_types = constraint_composition_type_maps[constraint.strip()]
                    for cat in cats:
                        cat_golden[cat].append(gl[k])
                        cat_pred[cat].append(pl[k])
                    for ct in comp_types:
                        comp_golden[ct].append(gl[k])
                        comp_pred[ct].append(pl[k])

            pointwise = calculate_pointwise_metrics(pred_labels, golden_labels)
            pairwise = calculate_pairwise_metrics(d["preference_graph"], scores)

            groupings = [
                ("model_type", get_model_type(d["response_generation_model"])),
                ("instruction_type", d["instruction_type"]),
                ("turn", get_turn(d)),
                ("constraint_count", c_bucket), 
            ]

            for group, key in groupings:
                collector.update(group, key, pointwise, pairwise, golden_labels, pred_labels)

        metrics = {}
        metrics["metrics_per_model_type"] = collector.compute_metrics("model_type")
        metrics["metrics_per_turn"] = collector.compute_metrics("turn")
        metrics["metrics_per_instruction_type"] = collector.compute_metrics("instruction_type")
        metrics["metrics_per_constraint_count"] = collector.compute_metrics("constraint_count")

        metrics["Overall"] = {}
        for dim in ["verification", "ranking"]:
            res = {}
            if dim == "verification":
                keys = ["Positive F1", "Negative F1"]
                raw_keys = ["pf1", "nf1"]
            else:
                keys = ["Accuracy", "Kendall Taub"]
                raw_keys = ["pair_acc", "kendall"]
            
            for k_out, k_raw in zip(keys, raw_keys):
                valid_scores = []
                for t_type in ["Single_Turn", "Multi_Turn", "System_Prompt"]:
                    if t_type in collector.stats["instruction_type"]: 
                        val = collector.get_raw_means("instruction_type", t_type, k_raw)
                        valid_scores.append(val)

                if valid_scores:
                    res[k_out] = sum(valid_scores) / len(valid_scores)
                else:
                    res[k_out] = 0
            metrics["Overall"][dim] = res


        metrics["metrics_per_constraint_category"] = {}
        if cat_golden:
            for cat in sorted(cat_golden.keys()):
                g, p = cat_golden[cat], cat_pred[cat]
                metrics["metrics_per_constraint_category"][cat] = {
                    "metrics": calculate_pointwise_metrics(p, g),
                    "MCC": calculate_mcc(g, p),
                    "label_distribution": {
                        "0": g.count(0), "1": g.count(1),
                        "ratio": g.count(0) / max(1, len(g))
                    }
                }


        metrics["metrics_per_constraint_composition_type"] = {}
        if comp_golden:
            for ct in sorted(comp_golden.keys()):
                g, p = comp_golden[ct], comp_pred[ct]
                metrics["metrics_per_constraint_composition_type"][ct] = {
                    "metrics": calculate_pointwise_metrics(p, g),
                    "MCC": calculate_mcc(g, p),
                    "label_distribution": {
                        "0": g.count(0), "1": g.count(1),
                        "ratio": g.count(0) / max(1, len(g))
                    }
                }

        with open(f"constraint_assessment_results/{file}", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)