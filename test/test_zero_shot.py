import sys
import os
sys.path.append(f"{os.environ['PROJ_DIR']}/model")

import torch
import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import analyze
from quant import *
from outlier import *
from eval import *
from collections import defaultdict
from pprint import pprint
from modelutils_llama import quantize_model_llama, reorder_model_llama, quantize_model_gptq_llama,  add_act_quant_wrapper_llama
from modelutils_opt import quantize_model_opt, reorder_model_opt, quantize_model_gptq_opt,  add_act_quant_wrapper_opt
from modelutils_mixtral import quantize_model_mixtral, add_act_quant_wrapper_mixtral, reorder_model_mixtral
from parallel_utils import map_layers_to_multi_gpus
from LMClass import LMClass
from eval import pattern_match
from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator
from datautils import *
import qLinearLayer
import pickle
import logging
from tqdm import tqdm

torch.set_printoptions(precision=10)
DEV = torch.device('cuda')

parser = argparse.ArgumentParser()

parser.add_argument(
    'model', type=str,
    help='LlaMa model to load; pass location of hugginface converted checkpoint.'
)
parser.add_argument(
    'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
    help='Where to extract calibration data from.'
)
parser.add_argument(
    '--seed',
    type=int, default=0, 
    help='Seed for sampling the calibration data.'
)
parser.add_argument(
    '--nsamples', type=int, default=128,
    help='Number of calibration data samples.'
)
# Quantization Method
parser.add_argument(
    '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
    help='#bits to use for quantizing weight; use 16 for evaluating base model.'
)
parser.add_argument(
    '--abits', type=int, default=16, choices=[2, 3, 4, 8, 16],
    help='#bits to use for quantizing activation; use 16 for evaluating base model.'
)
parser.add_argument(
    '--exponential', action='store_true',
    help='Whether to use exponent-only for weight quantization.'
)
parser.add_argument(
    '--a_sym', action='store_true',
    help='Whether to perform symmetric quantization. Default is asymmetric.'
)
parser.add_argument(
    '--w_sym', action='store_true',
    help='Whether to perform symmetric quantization. Default is asymmetric.'
)
parser.add_argument(
    '--static', action='store_true',
    help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
)
parser.add_argument(
    '--weight_group_size', type=int, default=0, choices=[0, 32, 64, 128, 256, 384, 768],
    help='Group size when quantizing weights. Using 128 as default quantization group.'
)
parser.add_argument( #- ??
    '--weight_channel_group', type=int, default=1,
    help='Group size of channels that will quantize together. (only for weights now)'
)
parser.add_argument(
    '--act_group_size', type=int, default=0, choices=[0, 64, 128, 256, 384, 768],
    help='Group size when quantizing activations. Using 128 as default quantization group.'
)
parser.add_argument(
    '--reorder', action='store_true',
    help='Whether to keep salient weight unquantized.'
)
parser.add_argument(
    '--act_sort_metric', type=str, default='hessian', choices=['abs_mean', 'hessian'],
    help='The metric used to sort the activations.'
)
parser.add_argument(
    '--keeper', type=int, default=0,
    help='Group size to keep outliers.'
)
parser.add_argument(
    '--keeper_precision', type=int, default=0, choices=[0, 1, 2, 3],
    help='Precision to keep outliers. 0 for FP16; 1 for E5M2; 2 for E4M3; 3 for INT8 Quant.'
)
parser.add_argument(
    '--cache_index', action='store_true',
    help='Whether to use cached reorder index'
)
parser.add_argument(
    '--tiling', type=int, default=0, choices=[0, 16],
    help='Tile-wise quantization granularity (Deprecated in Atom).'
)
parser.add_argument(
    '--kv_cache', action='store_true',
    help='Whether to quant KV_Cache'
)
parser.add_argument(
    '--use_gptq', action='store_true',
    help='Whether to use GPTQ for weight quantization.'
)
parser.add_argument(
    '--percdamp', type=float, default=.01,
    help='Percent of the average Hessian diagonal to use for dampening.'
)
parser.add_argument(
    '--a_clip_ratio', type=float, default=1.0,
    help='Clip ratio for activation quantization. new_max = max * clip_ratio'
)
parser.add_argument(
    '--w_clip_ratio', type=float, default=1.0,
    help='Clip ratio for weight quantization. new_max = max * clip_ratio'
)
parser.add_argument(
    '--kv_clip_ratio', type=float, default=1.0,
    help='Clip ratio for kv cache quantization. new_max = max * clip_ratio'
)
parser.add_argument(
    "--eval_ppl", action="store_true",
    help='Whether to evaluate perplexity.'
)
parser.add_argument(
    "--eval_common_sense", action="store_true",
    help='Whether to evaluate zero-shot accuray on commonsense reasoning tasks.'
)
parser.add_argument(
    "--multigpu", action="store_true", 
    help="at eval, map model to multiple gpus"
)
parser.add_argument(
    "--lm_eval_num_fewshot", type=int, default=0, 
    help="Number of shots in lm evaluation. Default is 0 for zero-shot."
)
parser.add_argument(
    "--lm_eval_limit", type=int, default=-1, 
    help="Limit the number of examples in lm evaluation"
)
parser.add_argument(
    '--save_dir', type=str, default='./saved',
    help='Path to store the reordering indices and quantized weights.'
)
parser.add_argument(
    '--quant_type', type=str, default='int', choices=['int', 'fp'],
    help='Determine the mapped data format by quant_type + n_bits. e.g. int8, fp4.'
)
parser.add_argument(
    '--save_model', action="store_true", help='Whether to save the quantized model.'
)
parser.add_argument(
    '--save_model_name', type=str, default='llama7b_quant_quantized.pth',
    help='Path to save the quantized model.'
)
parser.add_argument('--hook_data', action="store_true")
parser.add_argument('--test_name', type=str, default='llama2-7b_quant')

args = parser.parse_args()

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", args.test_name)
os.makedirs(RESULT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# stream
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# file
fh = logging.FileHandler(f"{RESULT_DIR}/log.txt", mode="w")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Running Zero-shot Evaluation")
logger.info(args)

model = torch.load(f"{args.save_dir}/{args.save_model_name}", map_location=torch.device('cpu'))

lm = LMClass(args, model)
lm.seqlen = 2048
lm.model.eval()
for param in lm.model.parameters():
    param.requires_grad = False

if args.multigpu:
    if ("llama" in args.model.lower()) or ("mixtral" in args.model.lower()):
        map_layers_to_multi_gpus(lm.model.model.layers)
        input_device = lm.model.model.layers[0].device
        output_device = lm.model.model.layers[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.model.embed_tokens.to(input_device)
        lm.model.model.norm.to(output_device)
        lm.model.lm_head.to(output_device)
    elif "opt" in args.model.lower():
        map_layers_to_multi_gpus(lm.model.model.decoder.layers)
        input_device = lm.model.model.decoder.layers[0].device
        output_device = lm.model.model.decoder.layers[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.model.decoder.embed_tokens.to(input_device)
        lm.model.model.decoder.embed_positions.to(input_device)
        lm.model.model.decoder.final_layer_norm.to(input_device)
        lm.model.lm_head.to(output_device)
else:
    lm._device = DEV
    lm.model = lm.model.to(lm.device)

# tasks = ["piqa","arc_easy","arc_challenge","boolq","hellaswag","winogrande"]
tasks = ["arc_challenge","boolq","hellaswag","winogrande"]
logger.info(f"Tasks: {tasks}")
hooked_data = {}
all_results = {}

def extract_exponent(tensor):
    tensor = tensor.to(torch.float16)
    bits = tensor.view(torch.int16)
    exponent_mask = 0x7f80
    exponent_bits = (bits & exponent_mask) >> 7
    exponent = exponent_bits - 127
    return exponent

def getDist(tensor):
  TargetInput = tensor
  IExps = extract_exponent(TargetInput)
  DistVal, DistCount = torch.unique(IExps, return_counts=True)
  InputDist = {}
  for val, count in zip(DistVal, DistCount):
    InputDist[val.item()] = count.item()
  return InputDist

def getDistBlocks(tensor, block_size):
  OriginShape = tensor.shape
  TargetInput = tensor.reshape(-1, block_size)
  IExps = extract_exponent(TargetInput)
  Dists = []
  # tqdm.write(f"TargetInput Shape: {TargetInput.shape}")
  # for i in tqdm(range(IExps.shape[0])):
  for i in range(IExps.shape[0]):
    DistVal, DistCount = torch.unique(IExps[i, :], return_counts=True)
    Dists.append((DistVal.cpu().numpy(), DistCount.cpu().numpy()))

  return Dists

try:
  for task in tasks:
    logger.info(f"Running Task: {task}")

    if args.hook_data:
      hooked_data[task] = {}
      hooks = []
      def stat_input_hook(name, module, input, output):
        if name not in hooked_data[task]:
          hooked_data[task][name] = []
        InputDist = getDistBlocks(input[0], 128)
        # WeightDist = getDistBlocks(module.weight, 128)
        hooked_data[task][name].append(InputDist)

      for name, m in model.model.named_modules():
          if isinstance(m, qLinearLayer.QLinearLayer):
              hooks.append(
                  m.register_forward_hook(functools.partial(stat_input_hook, name)) #- hook the function with name fixed
              )

    results = {}
    # tasks_str = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
    # tasks_str = "hellaswag,winogrande"
    # tasks_str = "hellaswag"
    tasks_str = task
    task_names = pattern_match(tasks_str.split(","), lm_tasks.ALL_TASKS)

    task_dict = lm_tasks.get_task_dict(task_names)
    t_results = lm_evaluator.evaluate(
        lm,
        task_dict,
        num_fewshot=args.lm_eval_num_fewshot,
        limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit
    )
    results.update(t_results)
    logger.info(f"Results: {results}")

    results_dict = results['results']
    all_results.update(results_dict)
    for task_name in tasks_str.split(','):
        if task_name in ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag']:
            logger.info(f"INFO {task_name} : {results_dict[task_name]['acc_norm']*100:.2f}")
        else:
            logger.info(f"INFO {task_name} : {results_dict[task_name]['acc']*100:.2f}")

    if args.hook_data:
      for hook in hooks:
          hook.remove()
except KeyboardInterrupt as e:
    logger.info("Keyboard Interrupted")
    logger.info(e)

if args.hook_data:
  pickle.dump(hooked_data, open(f"{RESULT_DIR}/hooked_data.pkl", "wb"))
pickle.dump(all_results, open(f"{RESULT_DIR}/all_results.pkl", "wb"))