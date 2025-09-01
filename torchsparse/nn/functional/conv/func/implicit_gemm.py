from typing import Dict

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse
import torchsparse.backend
import torchsparse.backends

__all__ = ["ImplicitGEMMConvolutionFuntion"]


class ImplicitGEMMConvolutionFuntion(Function):  # TorchSparse++
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        kmap: Dict,
        config: Dict,
        transposed: bool = False,
    ) -> torch.Tensor:
        sizes = kmap["sizes"]
        if not transposed:
            out_in_map = kmap["out_in_map"]
            reorder_out_in_map = kmap["reorder_out_in_map"]
            reduced_sorted_mask = kmap["reduced_sorted_mask"]
            reorder_loc = kmap["reorder_loc"]
            out_in_map_bwd = kmap["out_in_map_bwd"]
            # TODO: add reorder_out_in_map_bwd, reduced_sorted_mask_bwd_wgrad, reduced_sorted_mask_bwd_dgrad, reorder_loc_bwd
            # reorder_out_in_map_bwd = kmap["reorder_out_in_map_bwd"]
            # reduced_sorted_mask_bwd_wgrad = kmap["reduced_sorted_mask_bwd_wgrad"]
            # reduced_sorted_mask_bwd_dgrad = kmap["reduced_sorted_mask_bwd_dgrad"]
            # reorder_loc_bwd = kmap["reorder_loc_bwd"]
        else:
            out_in_map = kmap["out_in_map_t"]
            reorder_out_in_map = kmap["reorder_out_in_map_t"]
            reduced_sorted_mask = kmap["reduced_sorted_mask_t"]
            reorder_loc = kmap["reorder_loc_t"]
            out_in_map_bwd = kmap["out_in_map_bwd_t"]
            # TODO: add reorder_out_in_map_bwd, reduced_sorted_mask_bwd_wgrad, reduced_sorted_mask_bwd_dgrad, reorder_loc_bwd
            # reorder_out_in_map_bwd = kmap["reorder_out_in_map_bwd_t"]
            # reduced_sorted_mask_bwd_wgrad = kmap["reduced_sorted_mask_bwd_wgrad_t"]
            # reduced_sorted_mask_bwd_dgrad = kmap["reduced_sorted_mask_bwd_dgrad_t"]
            # reorder_loc_bwd = kmap["reorder_loc_bwd_t"]

        ifsort = config["ifsort"]

        input = input.contiguous()
        weight = weight.contiguous()
        if input.device.type != "cuda":
            if not transposed:
                output = torch.zeros(
                    sizes[1], weight.size(-1), dtype=input.dtype, device=input.device
                )
            else:
                # TODO(Haotian): ensure the original, upsampled size to be the same.
                output = torch.zeros(
                    sizes[0], weight.size(-1), dtype=input.dtype, device=input.device
                )

        if input.device.type == "cuda":
            if torch.float16 in [input.dtype, weight.dtype]:
                input = input.to(torch.float16)
                weight = weight.to(torch.float16)
            if torch.bfloat16 in [input.dtype, weight.dtype]:
                input = input.to(torch.bfloat16)
                weight = weight.to(torch.bfloat16)

            # input, weight, out_in_map, out_feats
            num_out_feats = sizes[1] if not transposed else sizes[0]
            num_out_channels = weight.shape[-1]

            if not ifsort:
                output = torchsparse.backend.conv_forward_implicit_gemm_cuda(
                    input,
                    weight,
                    out_in_map,
                    num_out_feats,
                    num_out_channels,
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                    torchsparse.backends.allow_bf16,
                )
            else:
                output = torchsparse.backend.conv_forward_implicit_gemm_sorted_cuda(
                    input,
                    weight,
                    reorder_out_in_map,
                    reduced_sorted_mask,
                    reorder_loc,
                    num_out_feats,
                    num_out_channels,
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                    torchsparse.backends.allow_bf16,
                )
        else:
            raise NotImplementedError
        ctx.save_for_backward(
            input,
            weight,
            out_in_map_bwd,
            # TODO: add reorder_out_in_map_bwd, reduced_sorted_mask_bwd_wgrad, reduced_sorted_mask_bwd_dgrad, reorder_loc_bwd
            # reorder_out_in_map_bwd,
            # reduced_sorted_mask_bwd_wgrad,
            # reduced_sorted_mask_bwd_dgrad,
            # reorder_loc_bwd,
        )
        ctx.transposed = transposed
        return output.to(weight.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        (
            input,
            weight,
            out_in_map_bwd,
            # TODO: add reorder_out_in_map_bwd, reduced_sorted_mask_bwd_wgrad, reduced_sorted_mask_bwd_dgrad, reorder_loc_bwd
            # reorder_out_in_map_bwd,
            # reduced_sorted_mask_bwd_wgrad,
            # reduced_sorted_mask_bwd_dgrad,
            # reorder_loc_bwd,
        ) = ctx.saved_tensors
        transposed = ctx.transposed

        grad_output = grad_output.contiguous()

        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        kernel_volume, ic, oc = weight.size()

        if grad_output.device.type == "cuda":
            # temporary fix for kernel volume < 32, workaround this to else branch
            if kernel_volume < 0:  # sort mode
                # dgrad
                grad_input = torchsparse.backend.conv_forward_implicit_gemm_sorted_cuda(
                    grad_output,
                    weight.transpose(2, 1).contiguous(),
                    reorder_out_in_map_bwd,
                    reduced_sorted_mask_bwd_dgrad,
                    reorder_loc_bwd,
                    input.size(0),
                    input.size(1),
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                    torchsparse.backends.allow_bf16
                )

                # wgrad
                grad_weight = (
                    (
                        torchsparse.backend.conv_backward_wgrad_implicit_gemm_sorted_cuda(
                            grad_output,
                            input,
                            reorder_out_in_map_bwd,
                            reduced_sorted_mask_bwd_wgrad,
                            reorder_loc_bwd,
                            1,
                            torchsparse.backends.allow_tf32,
                            torchsparse.backends.allow_fp16,
                            torchsparse.backends.allow_bf16
                        )
                    )
                    .reshape(kernel_volume, oc, ic)
                    .transpose(2, 1)
                    .contiguous()
                )

            else:  # unsort mode
                # dgrad
                grad_input = torchsparse.backend.conv_forward_implicit_gemm_cuda(
                    grad_output,
                    weight.transpose(2, 1).contiguous(),
                    out_in_map_bwd,
                    input.size(0),
                    input.size(1),
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                    torchsparse.backends.allow_bf16,
                )

                # wgrad
                grad_weight = (
                    (
                        torchsparse.backend.conv_backward_wgrad_implicit_gemm_cuda(
                            grad_output,
                            input,
                            out_in_map_bwd,
                            1,
                            torchsparse.backends.allow_tf32,
                            torchsparse.backends.allow_fp16,
                            torchsparse.backends.allow_bf16
                        )
                    )
                    .reshape(kernel_volume, oc, ic)
                    .transpose(2, 1)
                    .contiguous()
                )
        else:
            raise NotImplementedError
        return (grad_input, grad_weight, None, None, None)
