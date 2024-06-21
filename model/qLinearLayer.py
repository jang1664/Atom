import torch
import torch.nn as nn
from quant import fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_tensor, quantize_tensor_channel_group
import analyze

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
        
    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y
    
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
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
                saved_w = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

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
        
    @torch.no_grad()
    def forward(self, x):
        out_shape = [x.shape[0], x.shape[1], self.weight.shape[0]]
        weight_scales = analyze.get_weight_quant_scale(self.weight.cpu())
        output = torch.zeros(out_shape, device=x.device, dtype=x.dtype)

        for i in range(x.shape[0]):
          input_scales = analyze.get_input_qunat_scale(x.cpu())
          for k in range(0, x.shape[2], 128):
            input_chunk = x[i, :, k:k+128]
            input_scale = input_scales[:, k//128:k//128+1].to(x.device)
            weight_chunk = self.weight[:, k:k+128]
            weight_scale = weight_scales[:, k//128:k//128+1].to(x.device)

            quant_input = torch.round(input_chunk / input_scale).to(dtype=x.dtype)
            quant_weight = torch.round(weight_chunk / weight_scale).to(dtype=x.dtype)
            scale_mat = torch.matmul(input_scale, weight_scale.T)
            output[i, :, :] += (torch.functional.F.linear(quant_input, quant_weight, None) * scale_mat)

        # y = torch.functional.F.linear(x, self.weight, self.bias)
        # print(f"input shape : {x.shape}")
        # print(f"weight shape : {self.weight.shape}")
        # print(f"bias : {self.bias}")
        return output
    
    def to(self, *args, **kwargs):
        # print(type(self))
        # print(isinstance(self, QLinearLayerV2))
        super(QLinearLayerV2, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
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
                saved_w = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

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
        
    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        # print("hi i'm v2 qlinear")
        return y
    
    def to(self, *args, **kwargs):
        # print(type(self))
        # print(isinstance(self, QLinearLayerV2))
        super(QLinearLayerV2, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
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
                saved_w = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

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