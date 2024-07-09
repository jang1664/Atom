import torch
import torch.nn as nn
from quant import fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_tensor, quantize_tensor_channel_group
import gemm_acim

def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        args,
        enable_quant: bool = True
    ):
        super().__init__()
        self.args = args
        self.register_buffer('weight', originalLayer.weight)
        self.enable_quant = enable_quant # whether to allow quant on weights, default True
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        self.weight_maxq = None
        self.weight_scale = None
        self.weight_zero = None
        
    @torch.no_grad()
    def forward(self, x, x_int, x_scale):
        # print(x.shape) 
        # print(self.weight.shape)
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y
    
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.weight_scale is not None:
          self.weight_scale = self.weight_scale.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        if self.args.keeper > 0:
            saved_w = self.weight[:, -self.args.keeper:].clone().contiguous()
        
        # Whether to keep outliers in FP8
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            if self.args.keeper_precision == 1:
                saved_w = fake_quantize_quarter_E5M2(saved_w)
            elif self.args.keeper_precision == 2:
                saved_w = fake_quantize_quarter_E4M3(saved_w)
            elif self.args.keeper_precision == 3:
                saved_w, _, _ = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = 0

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling,
            quant_type=self.args.quant_type
        )

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = saved_w
            del saved_w
        return
    
    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.weight.device)
            self.weight = torch.index_select(self.weight, 1, in_reorder_index) #- weight was transposed
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.weight.device)
                self.weight = torch.index_select(self.weight, 0, out_reorder_index)
        return

class QLinearLayerV2(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.args = None
        self.register_buffer('weight', None)
        self.enable_quant = None # whether to allow quant on weights, default True
        self.bias = None
        self.weight_scale = None
        self.name = None
    
    @torch.no_grad()
    # def construct(self, origin_layer:QLinearLayer, kdim):
    def construct(self, origin_layer:QLinearLayer):
        # self.weight = origin_layer.weight[:, :kdim]
        self.weight = origin_layer.weight
        # self.weight_scale = origin_layer.weight_scale[:, :kdim//128]
        self.weight_scale = origin_layer.weight_scale
        self.bias = origin_layer.bias
        self.args = origin_layer.args
        self.enable_quant = origin_layer.enable_quant
        self.weight_int = torch.round(self.weight/torch.repeat_interleave(self.weight_scale, 128, dim=-1)).to(torch.int8)
        return self

    @torch.no_grad()
    def forward(self, x, x_int, x_scale):
        # x_dequant = x_int * torch.repeat_interleave(x_scale, 128, dim=-1)
        # err_cnt = torch.sum(torch.abs(x - x_dequant) > 1e-5)
        # if err_cnt > 0:
        #   print(f"{self.name} ACT Error count : {err_cnt}")
        
        # w_dequant = self.weight_int.to(torch.float16) * torch.repeat_interleave(self.weight_scale, 128, dim=-1)
        # err_cnt = torch.sum(torch.abs(self.weight - w_dequant) > 1e-5)
        # if err_cnt > 0:
        #   print(f"{self.name} WEIGHT Error count : {err_cnt}")

        # output = torch.functional.F.linear(x_dequant, w_dequant, None)

        out_shape = [x.shape[0], x.shape[1], self.weight.shape[0]]
        # weight_scales = analyze.get_weight_quant_scale(self.weight.cpu())
        # output = torch.zeros(out_shape, device=x.device, dtype=torch.float16)
        output = torch.zeros(out_shape, device=x.device, dtype=torch.float32)
        # output = torch.zeros(out_shape, device=x.device, dtype=torch.float64)

        for i in range(x.shape[0]):
          # input_scales = analyze.get_input_qunat_scale(x.cpu())
          for k in range(0, x.shape[2], 128):
            input_chunk = x_int[i, :, k:k+128]
            input_scale = x_scale[i, :, k//128:k//128+1]
            weight_chunk = self.weight_int[:, k:k+128]
            weight_scale = self.weight_scale[:, k//128:k//128+1]

            # input_dequant = input_chunk.to(torch.float64) * input_scale.to(torch.float64)
            # weight_dequant = weight_chunk.to(torch.float64) * weight_scale.to(torch.float64)
            # input_dequant = input_chunk.to(torch.float16) * input_scale
            # weight_dequant = weight_chunk.to(torch.float16) * weight_scale
            # output[i, :, :] += torch.functional.F.linear(input_dequant, weight_dequant)
            # output[i, :, :] += torch.functional.F.linear(input_dequant.to(torch), weight_dequant).to(torch.float64)
            # output[i, :, :] += torch.functional.F.linear(input_dequant.to(torch.float64), weight_dequant.to(torch.float64)).to(torch.float64)
            # output[i, :, :] += torch.functional.F.linear(input_dequant.to(torch.float32), weight_dequant.to(torch.float32)).to(torch.float32)

            # quant_input = torch.round(input_chunk / input_scale).to(dtype=x.dtype)
            # quant_weight = torch.round(weight_chunk / weight_scale).to(dtype=x.dtype)

            scale_mat = torch.functional.F.linear(input_scale.to(torch.float32), weight_scale.to(torch.float32))
            output[i, :, :] += (torch.functional.F.linear(input_chunk.to(torch.float32), weight_chunk.to(torch.float32)) * scale_mat)

            # output[i, :, :] += (torch.functional.F.linear(input_chunk.to(torch.float16), weight_chunk.to(torch.float16), None) * scale_mat).to(torch.float32)

        # # y = torch.functional.F.linear(x, self.weight, self.bias)
        # # print(f"input shape : {x.shape}")
        # # print(f"weight shape : {self.weight.shape}")
        # # print(f"bias : {self.bias}")

        return output.to(torch.float16)
    
    def to(self, *args, **kwargs):
        # print(type(self))
        # print(isinstance(self, QLinearLayerV2))
        super(QLinearLayerV2, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.weight_int = self.weight_int.to(*args, **kwargs)
        self.weight_scale = self.weight_scale.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        if self.args.keeper > 0:
            saved_w = self.weight[:, -self.args.keeper:].clone().contiguous()
        
        # Whether to keep outliers in FP8
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            if self.args.keeper_precision == 1:
                saved_w = fake_quantize_quarter_E5M2(saved_w)
            elif self.args.keeper_precision == 2:
                saved_w = fake_quantize_quarter_E4M3(saved_w)
            elif self.args.keeper_precision == 3:
                saved_w, _, _ = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = 0

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling,
            quant_type=self.args.quant_type
        )

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = saved_w
            del saved_w
        return
    
    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.weight.device)
            self.weight = torch.index_select(self.weight, 1, in_reorder_index) #- weight was transposed
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.weight.device)
                self.weight = torch.index_select(self.weight, 0, out_reorder_index)
        return

class QLinearLayerACIM(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.args = None
        self.register_buffer('weight', None)
        self.enable_quant = None # whether to allow quant on weights, default True
        self.bias = None
        self.weight_scale = None
        self.name = None
    
    @torch.no_grad()
    def construct(self, origin_layer:QLinearLayer):
        self.weight = origin_layer.weight
        self.weight_scale = origin_layer.weight_scale
        self.bias = origin_layer.bias
        self.args = origin_layer.args
        self.enable_quant = origin_layer.enable_quant
        self.weight_int = torch.round(self.weight/torch.repeat_interleave(self.weight_scale, 128, dim=-1)).to(torch.int8)
        return self

    @torch.no_grad()
    def forward(self, x, x_int, x_scale):
        # x_dequant = x_int * torch.repeat_interleave(x_scale, 128, dim=-1)
        # err_cnt = torch.sum(torch.abs(x - x_dequant) > 1e-5)
        # if err_cnt > 0:
        #   print(f"{self.name} ACT Error count : {err_cnt}")
        
        # w_dequant = self.weight_int.to(torch.float16) * torch.repeat_interleave(self.weight_scale, 128, dim=-1)
        # err_cnt = torch.sum(torch.abs(self.weight - w_dequant) > 1e-5)
        # if err_cnt > 0:
        #   print(f"{self.name} WEIGHT Error count : {err_cnt}")
        # print(f"{self.name} Run")
        # print(f"input shape : {x.shape}")
        # print(f"weight shape : {self.weight.shape}")
        # print(x_int.dtype)
        # print(x_scale.dtype)
        # print(self.weight_int.dtype)
        # print(self.weight_scale.dtype)
        out_shape = [x.shape[0], x.shape[1], self.weight.shape[0]]
        output = torch.zeros(out_shape, device=x.device, dtype=torch.float32)

        for i in range(x_int.shape[0]):
          output[i, :, :] = gemm_acim.forward(
              x_int[i, :, :].to(torch.uint8).to(x.device),
              self.weight_int.to(torch.uint8).to(x.device),
              x_scale[i, :, :].to(torch.float32).to(x.device),
              self.weight_scale.to(torch.float32).to(x.device),
              4, 4, True).reshape([x.shape[1], self.weight.shape[0]])

        return output.to(torch.float16)
    
    def to(self, *args, **kwargs):
        super(QLinearLayerACIM, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.weight_int = self.weight_int.to(*args, **kwargs)
        self.weight_scale = self.weight_scale.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        if self.args.keeper > 0:
            saved_w = self.weight[:, -self.args.keeper:].clone().contiguous()
        
        # Whether to keep outliers in FP8
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            if self.args.keeper_precision == 1:
                saved_w = fake_quantize_quarter_E5M2(saved_w)
            elif self.args.keeper_precision == 2:
                saved_w = fake_quantize_quarter_E4M3(saved_w)
            elif self.args.keeper_precision == 3:
                saved_w, _, _ = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = 0

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling,
            quant_type=self.args.quant_type
        )

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = saved_w
            del saved_w
        return
    
    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.weight.device)
            self.weight = torch.index_select(self.weight, 1, in_reorder_index) #- weight was transposed
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.weight.device)
                self.weight = torch.index_select(self.weight, 0, out_reorder_index)
        return