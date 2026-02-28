
# IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation

This repository is the official implementation of [IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](https://arxiv.org/abs/2511.01014). 

## Data Examples

- We provide the data of IF-RewardBench in ```data/```
-  The format of an example is as follows.

  - `id` (integer): A unique identifier for this example.
  - `response_generation_model` (string): The response generation model of this example.
  - `instruction_type` (string): The instruction type of this example.
  - `messages` (list): The user instruction, which may contain the system prompt and conversation history.
  - `checklist` (list): The constraint checklist for the user instruction.
  - `constraint_type` (list): The constraint categories and constraint composition types in each constraint of the checklist.
  - `responses` (list): All responses and their corresponding instruction-following judgements for each constraint.
  - `preference_graph` (list): All preference relations among these responses.

## Judge Model Inference

- We provide the codes for the judge model inference on constraint assessment and overall assessment in ```inference/```, which are based on vllm framework
- As we have randomly shuffled the positions of candidate responses in pairwise comparison, we place this position information of **the above data examples** in ```inference/position_maps_examples.json```
- We also place the inference results of QwQ-32B in **the above data examples** in ```inference/constraint_assessment_results``` and ```inference/overall_assessment_results```

## Evaluation Metrics Calculation

- We provide the codes for the evaluation metrics calculation on constraint assessment and overall assessment in ```metrics/```
- We also place the evaluation results of QwQ-32B in **the above data examples** in ```metrics/constraint_assessment_results``` and ```metrics/overall_assessment_results```

