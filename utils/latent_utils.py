import warnings
import torch
import torch.nn.functional as F
from peft.utils.other import transpose

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def get_delta_weight(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = transpose(
        weight_B @ self.default_lora_latent_mapping.weight @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor


def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result

# def transpose(weight, fan_in_fan_out):
#     if not fan_in_fan_out:
#         return weight

#     if isinstance(weight, torch.nn.Parameter):
#         return torch.nn.Parameter(weight.T)
#     return weight.T

# def get_weight_norm(weight, lora_weight, scaling) -> torch.Tensor:
#     # calculate L2 norm of weight matrix, column-wise
#     weight = transpose(weight, self.fan_in_fan_out)
#     weight = weight + scaling * lora_weight
#     weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
#     return weight_norm

def forward_edora(self, x:torch.Tensor):

    # Maybe we can use the following instead:
    # result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    result = self.base_layer(x)
    
    torch_result_dtype = result.dtype
    lora_A = self.lora_A[self.active_adapter[0]]
    lora_B = self.lora_B[self.active_adapter[0]]
    lora_R = self.default_lora_latent_mapping
    dropout = self.lora_dropout[self.active_adapter[0]]
    scaling = self.scaling[self.active_adapter[0]]
    x = x.to(lora_A.weight.dtype)
    x = dropout(x)

    # Applying EDoRA
    lora_weight = lora_B.weight @ lora_R.weight @ lora_A.weight
    magnitude = self.lora_magnitude_vector[self.active_adapter[0]]
    weight = self.get_base_layer().weight
    
    weight = weight.to(x.dtype)
    weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
    # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
    # "[...] we suggest treating ||V +∆V ||_c in
    # Eq. (5) as a constant, thereby detaching it from the gradient
    # graph. This means that while ||V + ∆V ||_c dynamically
    # reflects the updates of ∆V , it won’t receive any gradient
    # during backpropagation"
    weight_norm = weight_norm.detach()
    mag_norm_scale = (magnitude / weight_norm).view(1, -1)
    result_dora = (mag_norm_scale - 1) * (
        F.linear(x, transpose(weight, self.fan_in_fan_out))
    ) + mag_norm_scale * lora_B(lora_R(lora_A(x))) * scaling

    # Note: Computation could potentially be accelerated by using the code below instead of calculating X@W again.
    # This is only correct if dropout=0, otherwise results will differ:
    # https://github.com/huggingface/peft/pull/1474#issuecomment-1964682771
    # bias = self.get_base_layer().bias
    # if bias is not None:
    #     result = result - bias
    # result = mag_norm_scale * result + mag_norm_scale * lora_B(lora_A(x)) * scaling
    # if bias is not None:
    #     result = result + bias

    result = result + result_dora
    result = result.to(torch_result_dtype)

    return result

