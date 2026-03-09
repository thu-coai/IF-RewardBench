# IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation

This repository is the official implementation of [IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](https://arxiv.org/abs/2603.04738). 

IF-RewardBench is a comprehensive meta-evaluation benchmark to assess the capability of judge models in instruction-following evaluation. It comprises 842 diverse instructions covering single-turn interaction, multi-turn interaction, and system-prompt steerability scenarios. For each instruction, our benchmark constructs a preference graph containing all pairwise preferences among multiple responses based on instruction-following quality, enabling a listwise evaluation paradigm to assess the ranking capabilities of judge models.

---

## 📁 Data Format

The data of IF-RewardBench is provided in `data/`. The format of an example is as follows:

* **`id`** *(integer)*: A unique identifier for this example.
* **`response_generation_model`** *(string)*: The response generation model of this example.
* **`instruction_type`** *(string)*: The instruction type of this example.
* **`messages`** *(list)*: The user instruction, which may contain the system prompt and conversation history.
* **`checklist`** *(list)*: The constraint checklist for the user instruction.
* **`constraint_type`** *(list)*: The constraint categories and constraint composition types in each constraint of the checklist.
* **`responses`** *(list)*: All responses and their corresponding instruction-following judgements for each constraint.
* **`preference_graph`** *(list)*: All preference relations among these responses.

---

## 🚀 Judge Model Inference

We provide the codes for the judge model inference on constraint assessment and overall assessment in `inference/`, which are based on the vLLM framework. 

**Constraint Assessment**

```shell
cd inference/
python constraint_assessment_inference_vllm.py --model_name <model_name> --model_path <model_path>
```

**Overall Assessment**

```shell
cd inference/
python overall_assessment_inference_vllm.py --model_name <model_name> --model_path <model_path>
```

As we have randomly shuffled the positions of candidate responses in pairwise comparison, we place this position information in `inference/position_maps.json`.

---

## 📊 Evaluation Metrics Calculation

We provide the codes for the evaluation metrics calculation on constraint assessment and overall assessment in `metrics/`.

**Constraint Assessment**

```shell
cd metrics/
python analysis_constraint_assessment.py
```

**Overall Assessment**

```shell
cd metrics/
python analysis_overall_assessment.py
```

## 👏 Citation

```
@article{wen2026if,
  title={IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation},
  author={Wen, Bosi and Niu, Yilin and Wang, Cunxiang and Ling, Xiaoying and Zhang, Ying and Ke, Pei and Wang, Hongning and Huang, Minlie},
  journal={arXiv preprint arXiv:2603.04738},
  year={2026}
}
```

Please kindly cite our paper if this paper and the codes are helpful.