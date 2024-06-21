import torch
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
from tqdm import tqdm

import gc
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from qLinearLayer import find_qlinear_layers
from qLlamaLayer import QLlamaDecoderLayer
from qMixtralLayer import QMixtralDecoderLayer
from gptq import GPTQ, Quantizer_GPTQ
from functools import partial

from quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper

import argparse
import os

def load_model(args):
  def skip(*args, **kwargs):
      pass
  torch.nn.init.kaiming_uniform_ = skip
  torch.nn.init.uniform_ = skip
  torch.nn.init.normal_ = skip
  from transformers import LlamaForCausalLM
  model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
  model.seqlen = 2048
  get_act_stats_func = get_act_stats_llama #- get linear layer's input and output column-wises L2 norm
  reorder_model_func = reorder_model_llama #- reorder weight and register buffer for act reorder index
  add_act_quant_wrapper_func = add_act_quant_wrapper_llama #- configure activation quantization and attach
  quantize_model_gptq_func = quantize_model_gptq_llama
  quantize_model_func = quantize_model_llama
  eval_func = llama_eval
  DEV = torch.device('cuda:0')

  model_name = args.model.lower().split('/')[-1]
  index_filename = f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt'
  assert os.path.isfile(index_filename), "reorder index file not found."
  print("Loading cached reording index from disk...")
  reorder_index = torch.load(index_filename)

  model.config.use_cache = False
  layers = model.model.layers
  for i in tqdm(range(len(layers))):
      if isinstance(layers[i], LlamaDecoderLayer):
        m = QLlamaDecoderLayer(
            originalLayer=layers[i],
            args=args, #- command line args
        )
      
        nameTemplate = 'layers.{}.{}.{}.{}' # Something like layers.10.self_attn.q_proj

        #- reorder the weights for MLP
        m.input_layernorm.register_buffer('reorder_index', #- for later fusion of input reordering
            reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')] # Random choose one from k,q,v proj.
        )
        m.post_attention_layernorm.register_buffer('reorder_index',
            reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')]
        )
        m.self_attn.register_buffer('reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')])

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu() #- replace llama decoder layer
        del m
        torch.cuda.empty_cache()

  scales = defaultdict(lambda: None)
  model = add_act_quant_wrapper_func(model, device=DEV, args=args, scales=scales) #- activation is dynamic quantization

  model_state_dict = torch.load("./saved/llama2-7b_quantized.pt")
  model.load_state_dict(model_state_dict)
  model.eval()

  return model

def get_weight_quant_scale(weight):
  # weight = weight.float()
  weight = weight
  shape = weight.shape
  normal_scales = torch.zeros((shape[0]//2, (shape[1]-128)//128), dtype=torch.float16)

  #- normal value
  for i in range(0, (shape[1]-128), 128):
    start_idx = i
    end_idx = i+128
    weight_block = weight[:, start_idx:end_idx]
    weight_block = weight_block.reshape(weight_block.shape[0]//2, -1)

    min_val = weight_block.min(dim=1)[0].to(dtype=torch.float16)
    max_val = weight_block.max(dim=1)[0].to(dtype=torch.float16)

    pos_max_idx = (torch.abs(max_val) >= torch.abs(min_val))
    normal_scales[pos_max_idx, i//128] = (max_val[pos_max_idx].abs()/7).to(dtype=torch.float16)
    normal_scales[~pos_max_idx, i//128] = (min_val[~pos_max_idx].abs()/8).to(dtype=torch.float16)
  
  expanded_normal_scales = torch.zeros((shape[0], (shape[1]-128)//128), dtype=torch.float16)
  for i in range(normal_scales.shape[0]):
    expanded_normal_scales[2*i] = normal_scales[i]
    expanded_normal_scales[2*i+1] = normal_scales[i]

  #- outlier
  out_scales = torch.zeros(shape[0], 1, dtype=torch.float16)
  start_idx = shape[1]-128
  end_idx = shape[1]
  weight_block = weight[:, start_idx:end_idx]

  min_val = weight_block.min(dim=1)[0].to(dtype=torch.float16)
  max_val = weight_block.max(dim=1)[0].to(dtype=torch.float16)

  pos_max_idx = (torch.abs(max_val) >= torch.abs(min_val))
  out_scales[pos_max_idx, 0] = max_val[pos_max_idx].abs()/(127)
  out_scales[~pos_max_idx, 0] = min_val[~pos_max_idx].abs()/(128)

  scales = torch.cat([expanded_normal_scales, out_scales], dim=-1)
  
  return scales

def check_normal_scale(weight, scales, row, col):
  weight_values = weight[:, 128*(col):128*(col+1)].reshape(weight.shape[0]//2, -1)[row,:]
  weight_value_set = set(torch.unique(weight_values).tolist())
  weight_value_set = sorted(weight_value_set, key=lambda x: abs(x))
  print(weight_value_set)
  print(scales[row,col])

def check_out_scale(weight, scales, row):
  start_idx = weight.shape[1]-128
  weight_values = weight[row, start_idx:]
  weight_value_set = set(torch.unique(weight_values).tolist())
  weight_value_set = sorted(weight_value_set, key=lambda x: abs(x))
  print(weight_value_set)
  print(scales[row])

@torch.no_grad()
def extract_io(model, testenc, dev):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    inputs = {}
    outputs = {}

    def get_tensors_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        
        if f"{name}.input" not in inputs:
            inputs[f"{name}.input"] = [x.cpu()]
        else:
          if len(inputs[f"{name}.input"]) < 3:
            inputs[f"{name}.input"].append(x.cpu())
        
        if f"{name}.output" not in outputs:
            outputs[f"{name}.output"] = [y.cpu()]
        else:
          if len(outputs[f"{name}.output"]) < 3:
            outputs[f"{name}.output"].append(y.cpu())

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, QLinearLayer):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(get_tensors_hook, name=name))
            )

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            if(j == 2):
               break
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    for hook in hooks:
      hook.remove()

    gc.collect()
    return inputs, outputs

# def get_qunat_input(input_tensor):
#     group_size = 128
#     out_num = 128
#     non_out_n_bits = 4
#     non_out_clip_ratio = 0.9

#     q_max = (2**(non_out_n_bits-1)-1)
#     q_min = (-2**(non_out_n_bits-1))

#     input_tensor = input_tensor.squeeze(0)

#     savedShape = input_tensor.shape
#     non_outlier = input_tensor[:,:-out_num]

#     non_outlier = non_outlier.reshape(-1, group_size)

#     w_max = non_outlier.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
#     w_max = w_max * non_out_clip_ratio

#     scales = w_max / q_max
#     base = torch.zeros_like(scales)

#     quantized_value = torch.clamp(torch.round(non_outlier / scales) + base, q_min, q_max)
#     dequantized_value = (quantized_value - base) * scales

#     non_out_scales = scales.reshape(savedShape[0], -1)
#     non_out_base = base.reshape(savedShape[0], -1)
#     non_out_quantized_value = quantized_value.reshape(savedShape[0], -1)
#     non_out_dequantized_value = dequantized_value.reshape(savedShape[0], -1)

#     out_n_bits = 8
#     out_clip_ratio = 1.0
#     outlier = input_tensor[:,-out_num:]
#     q_max = (2**(out_n_bits-1)-1)
#     q_min = (-2**(out_n_bits-1))

#     w_max = outlier.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
#     w_max = w_max * out_clip_ratio

#     scales = w_max / q_max
#     base = torch.zeros_like(scales)

#     quantized_value = torch.clamp(torch.round(outlier / scales) + base, q_min, q_max)
#     dequantized_value = (quantized_value - base) * scales

#     out_scales = scales.reshape(savedShape[0], -1)
#     out_base = base.reshape(savedShape[0], -1)
#     out_quantized_value = quantized_value.reshape(savedShape[0], -1)
#     out_dequantized_value = dequantized_value.reshape(savedShape[0], -1)

#     scales = torch.cat([non_out_scales, out_scales], dim=-1)
#     bases = torch.cat([non_out_base, out_base], dim=-1)
#     quantized_value = torch.cat([non_out_quantized_value, out_quantized_value], dim=-1)
#     dequantized_value = torch.cat([non_out_dequantized_value, out_dequantized_value], dim=-1)

#     return quantized_value, scales, bases, dequantized_value

def get_input_qunat_scale(input_tensor):
  # input_tensor = input_tensor.float()
  input_tensor = input_tensor.squeeze(0)
  shape = input_tensor.shape
  normal_scales = torch.zeros((shape[0], (shape[1]-128)//128), dtype=torch.float16)

  #- normal value
  for i in range(0, (shape[1]-128), 128):
    start_idx = i
    end_idx = i+128
    input_tensor_block = input_tensor[:, start_idx:end_idx]

    min_val = input_tensor_block.min(dim=1)[0].to(dtype=torch.float16)
    max_val = input_tensor_block.max(dim=1)[0].to(dtype=torch.float16)

    pos_max_idx = (torch.abs(max_val) >= torch.abs(min_val))
    normal_scales[pos_max_idx, i//128] = (max_val[pos_max_idx].abs()/7).to(dtype=torch.float16)
    normal_scales[~pos_max_idx, i//128] = (min_val[~pos_max_idx].abs()/8).to(dtype=torch.float16)
  
  #- outlier
  out_scales = torch.zeros(shape[0], 1, dtype=torch.float16)
  start_idx = shape[1]-128
  end_idx = shape[1]
  input_tensor_block = input_tensor[:, start_idx:end_idx]

  min_val = input_tensor_block.min(dim=1)[0].to(dtype=torch.float16)
  max_val = input_tensor_block.max(dim=1)[0].to(dtype=torch.float16)

  pos_max_idx = (torch.abs(max_val) >= torch.abs(min_val))
  out_scales[pos_max_idx, 0] = max_val[pos_max_idx].abs()/(127)
  out_scales[~pos_max_idx, 0] = min_val[~pos_max_idx].abs()/(128)

  scales = torch.cat([normal_scales, out_scales], dim=-1)
  
  return scales

def get_input_quant_param_dict(inputs):
  quantized_inputs = {}
  scales = {}
  bases = {}
  dequantized_inputs = {}
  for name, tensor_list in inputs.items():
    quantized_inputs[f"{name}.quantized_input"] = []
    scales[f"{name}.scale"] = []
    bases[f"{name}.base"] = []
    dequantized_inputs[f"{name}.dequantized_input"] = []
    for i, tensor in enumerate(tensor_list):
      # quantized_value, scale, base, dequantized_value = get_qunat_input(tensor)
      scale = get_input_qunat_scale(tensor)
      # quantized_inputs[f"{name}.quantized_input"].append(quantized_value.cpu())
      scales[f"{name}.scale"].append(scale.cpu())
      # bases[f"{name}.base"].append(base.cpu())
      # dequantized_inputs[f"{name}.dequantized_input"].append(dequantized_value.cpu())
      break
  
  return scales

def get_weight_quant_param_dict(model):
  scales = {}
  for name, m in model.model.named_modules():
      if isinstance(m, QLinearLayer):
        ns = get_weight_quant_scale(m.weight.cpu())
        scales[f"{name}.scale"] = ns
  
  return scales

def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)