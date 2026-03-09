# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.utils.vn_entropy import VNEntropyCalculator
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        self.vn_entropy_calculator = None
        self.vn_entropy_pca_dim = None
        self.vn_entropy_top_k = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_logits: bool = False,
        disable_inplace_backward: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if return_logits is True:
                    logits: (bs, response_len, vocab_size)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # Keep log-prob computation path aligned across call sites when requested.
                    inplace_backward = not (calculate_entropy or disable_inplace_backward)
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if return_logits and not self.use_fused_kernels:
                        logits_rmpad = gather_outputs_and_unpad(
                            logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if return_logits and not self.use_fused_kernels:
                        logits_rmpad = logits_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if return_logits and not self.use_fused_kernels:
                    full_logits = pad_input(
                        hidden_states=logits_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                if return_logits and not self.use_fused_kernels:
                    logits = full_logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if return_logits and not self.use_fused_kernels:
                outputs["logits"] = logits
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    def _get_embedding_matrix(self) -> torch.Tensor:
        """Extract embedding matrix from model, handling FSDP sharding."""
        model = self.actor_module
        param_ctx = nullcontext()

        if isinstance(model, FSDP):
            param_ctx = FSDP.summon_full_params(model, writeback=False, recurse=False)

        with param_ctx:
            base_model = getattr(model, "_fsdp_wrapped_module", model)
            if hasattr(base_model, "module"):
                base_model = base_model.module
            if hasattr(base_model, "get_input_embeddings"):
                embeddings = base_model.get_input_embeddings()
            elif hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
                embeddings = base_model.model.embed_tokens
            elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "wte"):
                embeddings = base_model.transformer.wte
            else:
                raise ValueError("Unsupported model for embedding extraction.")

            weight = embeddings.weight
            if hasattr(weight, "full_tensor"):
                weight = weight.full_tensor()

            # Clone within FSDP context (tensor becomes invalid after exiting)
            return weight.detach().clone()

    def update_vn_entropy_pca(self, pca_dim: int = 64, top_k: int = 64):
        """Initialize or update VN entropy calculator with new PCA projection."""
        logger.info(f"[dp_actor] update_vn_entropy_pca: START (pca_dim={pca_dim}, top_k={top_k})")
        embedding_matrix = self._get_embedding_matrix()
        logger.info(f"[dp_actor] update_vn_entropy_pca: got embedding_matrix {embedding_matrix.shape}")
        self.vn_entropy_pca_dim = pca_dim
        self.vn_entropy_top_k = top_k
        if self.vn_entropy_calculator is None:
            logger.info("[dp_actor] update_vn_entropy_pca: creating new VNEntropyCalculator")
            self.vn_entropy_calculator = VNEntropyCalculator(
                embedding_matrix=embedding_matrix,
                pca_dim=pca_dim,
                top_k=top_k,
            )
            logger.info("[dp_actor] update_vn_entropy_pca: VNEntropyCalculator created")
        else:
            logger.info("[dp_actor] update_vn_entropy_pca: updating existing calculator")
            self.vn_entropy_calculator.top_k = top_k
            self.vn_entropy_calculator.update_pca(embedding_matrix)
            logger.info("[dp_actor] update_vn_entropy_pca: calculator updated")

    def compute_vn_entropy(self, data: DataProto) -> torch.Tensor:
        """Compute VN entropy for all positions in the batch."""
        logger.info("[dp_actor] compute_vn_entropy: START")
        if self.vn_entropy_calculator is None:
            pca_dim = data.meta_info["vn_entropy_pca_dim"]
            top_k = data.meta_info["vn_entropy_top_k"]
            logger.info(f"[dp_actor] compute_vn_entropy: initializing calculator with pca_dim={pca_dim}, top_k={top_k}")
            self.update_vn_entropy_pca(pca_dim=pca_dim, top_k=top_k)

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info["pad_token_id"]
        chunk_size = data.meta_info["vn_entropy_chunk_size"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        logger.info(f"[dp_actor] compute_vn_entropy: processing {len(micro_batches)} micro_batches")
        vn_entropy_lst = []
        for i, micro_batch in enumerate(micro_batches):
            logger.info(f"[dp_actor] compute_vn_entropy: micro_batch {i}/{len(micro_batches)} - moving to device")
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            logger.info(f"[dp_actor] compute_vn_entropy: micro_batch {i}/{len(micro_batches)} - forward pass START")
            with torch.no_grad():
                if self.use_fused_kernels:
                    original_use_fused_kernels = self.use_fused_kernels
                    self.use_fused_kernels = False
                    try:
                        outputs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, return_logits=True
                        )
                    finally:
                        self.use_fused_kernels = original_use_fused_kernels
                else:
                    outputs = self._forward_micro_batch(model_inputs, temperature=temperature, return_logits=True)
                logits = outputs["logits"]
                del outputs
                logger.info(
                    f"[dp_actor] compute_vn_entropy: micro_batch {i}/{len(micro_batches)} - "
                    f"computing VN entropy for logits {logits.shape}"
                )
                vn_entropy = self.vn_entropy_calculator.compute_vn_entropy(logits=logits, chunk_size=chunk_size)
                del logits
            logger.info(f"[dp_actor] compute_vn_entropy: micro_batch {i}/{len(micro_batches)} - VN entropy computed")
            vn_entropy_lst.append(vn_entropy.cpu())

        logger.info("[dp_actor] compute_vn_entropy: concatenating results")
        vn_entropy = torch.concat(vn_entropy_lst, dim=0)
        if use_dynamic_bsz:
            vn_entropy = restore_dynamic_batch(vn_entropy, batch_idx_list)
        logger.info(f"[dp_actor] compute_vn_entropy: DONE, output shape {vn_entropy.shape}")
        return vn_entropy

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info["pad_token_id"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info["pad_token_id"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # Include grad_sq for scheduled ECA per-step reweighting
        if "grad_sq" in data.batch.keys():
            select_keys.append("grad_sq")
        # Include rollout Shannon entropy for per-step KL entropy splits
        if "rollout_entropy" in data.batch.keys():
            select_keys.append("rollout_entropy")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Extract scheduled ECA gamma schedule before split (meta_info survives select/split)
        eca_softmax_gamma_schedule = data.meta_info.get("eca_softmax_gamma_schedule", None)

        # Extract on-policy ECA softmax config (recompute grad_sq from current policy each micro-batch)
        eca_softmax_on_policy_grad_sq = data.meta_info.get("eca_softmax_on_policy_grad_sq", False)
        eca_softmax_gamma = data.meta_info.get("eca_softmax_gamma", 0.0)

        # Extract entropy-top config (train only top X% highest-entropy tokens)
        entropy_top = data.meta_info.get("entropy_top", False)
        entropy_top_ratio = data.meta_info.get("entropy_top_ratio", 0.2)

        # Align pre-step metric recomputation with old_log_probs generation settings.
        metric_use_dynamic_bsz = data.meta_info.get("old_log_prob_use_dynamic_bsz", self.config.use_dynamic_bsz)
        metric_max_token_len_per_gpu = data.meta_info.get(
            "old_log_prob_max_token_len_per_gpu", self.config.ppo_max_token_len_per_gpu
        )
        metric_micro_batch_size_per_gpu = data.meta_info.get(
            "old_log_prob_micro_batch_size_per_gpu", self.config.ppo_micro_batch_size_per_gpu
        )

        # Shuffle within this GPU's shard before splitting into mini-batches.
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        # Shuffle mini-batch order to remove systematic sequence-length ordering
        # from _balance_batch (which sorts in an inverted-U pattern).
        # Seed must be identical across FSDP ranks (so they stay in sync) but
        # vary across RL steps (so each step gets a different permutation).
        import random

        if not hasattr(self, "_shuffle_step"):
            self._shuffle_step = 0
        self._shuffle_step += 1
        random.Random(42 + self._shuffle_step).shuffle(mini_batches)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        def _clone_metric_mini_batch(source_mini_batch: DataProto) -> DataProto:
            metric_mini_batch = source_mini_batch.select(deepcopy=True)
            for key in metric_mini_batch.batch.keys():
                metric_mini_batch.batch[key] = metric_mini_batch.batch[key].clone()
            return metric_mini_batch

        def _compute_entropy_threshold(target_mini_batch: DataProto):
            if "rollout_entropy" not in target_mini_batch.batch.keys():
                return None

            entropy = target_mini_batch.batch["rollout_entropy"]
            response_mask = target_mini_batch.batch["response_mask"]
            valid_entropy = entropy[response_mask.bool()]
            if valid_entropy.numel() == 0:
                return None
            return torch.quantile(valid_entropy.float(), 0.8)

        # Fixed first/last update mini-batches for measuring policy drift over the PPO pass.
        first_metric_mini_batch = _clone_metric_mini_batch(mini_batches[0])
        last_metric_mini_batch = _clone_metric_mini_batch(mini_batches[-1])
        first_entropy_threshold = _compute_entropy_threshold(first_metric_mini_batch)
        last_entropy_threshold = _compute_entropy_threshold(last_metric_mini_batch)

        def _split_metric_micro_batches(target_mini_batch: DataProto):
            if metric_use_dynamic_bsz:
                return prepare_dynamic_batch(
                    target_mini_batch,
                    max_token_len=metric_max_token_len_per_gpu * self.ulysses_sequence_parallel_size,
                )[0]
            return target_mini_batch.split(metric_micro_batch_size_per_gpu)

        def _compute_prestep_k3_metrics(
            target_mini_batch: DataProto, entropy_threshold, mode: str, use_self_baseline: bool = False
        ):
            with torch.no_grad():
                was_training = self.actor_module.training
                if mode == "eval":
                    self.actor_module.eval()
                elif mode == "train":
                    self.actor_module.train()
                else:
                    raise ValueError(f"Unknown KL eval mode: {mode}")

                kl_sum = 0.0
                kl_count = 0
                kl_low_sum = 0.0
                kl_low_count = 0
                kl_high_sum = 0.0
                kl_high_count = 0

                metric_micro_batches = _split_metric_micro_batches(target_mini_batch)

                for metric_mb in metric_micro_batches:
                    metric_mb = metric_mb.to(get_device_id())
                    metric_inputs = {**metric_mb.batch, **metric_mb.non_tensor_batch, "pad_token_id": pad_token_id}
                    metric_outputs = self._forward_micro_batch(
                        metric_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        disable_inplace_backward=True,
                    )
                    metric_log_prob = metric_outputs["log_probs"]
                    metric_old_lp = metric_log_prob.detach() if use_self_baseline else metric_inputs["old_log_probs"]
                    metric_rmask = metric_inputs["response_mask"]
                    metric_kl = kl_penalty(logprob=metric_old_lp, ref_logprob=metric_log_prob, kl_penalty="k3")

                    kl_sum += (metric_kl * metric_rmask).sum().item()
                    kl_count += metric_rmask.sum().item()

                    if entropy_threshold is not None and "rollout_entropy" in metric_inputs:
                        metric_ent = metric_inputs["rollout_entropy"]
                        low_mask = (metric_ent <= entropy_threshold).float() * metric_rmask
                        high_mask = (metric_ent > entropy_threshold).float() * metric_rmask
                        kl_low_sum += (metric_kl * low_mask).sum().item()
                        kl_low_count += low_mask.sum().item()
                        kl_high_sum += (metric_kl * high_mask).sum().item()
                        kl_high_count += high_mask.sum().item()

                if was_training:
                    self.actor_module.train()
                else:
                    self.actor_module.eval()

                return {
                    "all": (kl_sum / kl_count) if kl_count > 0 else None,
                    "low": (kl_low_sum / kl_low_count) if kl_low_count > 0 else None,
                    "high": (kl_high_sum / kl_high_count) if kl_high_count > 0 else None,
                }

        def _compute_prestep_is_ratio_mse_metrics(
            target_mini_batch: DataProto, entropy_threshold, mode: str, use_self_baseline: bool = False
        ):
            with torch.no_grad():
                was_training = self.actor_module.training
                if mode == "eval":
                    self.actor_module.eval()
                elif mode == "train":
                    self.actor_module.train()
                else:
                    raise ValueError(f"Unknown IS eval mode: {mode}")

                mse_sum = 0.0
                mse_count = 0
                mse_low_sum = 0.0
                mse_low_count = 0
                mse_high_sum = 0.0
                mse_high_count = 0

                metric_micro_batches = _split_metric_micro_batches(target_mini_batch)

                for metric_mb in metric_micro_batches:
                    metric_mb = metric_mb.to(get_device_id())
                    metric_inputs = {**metric_mb.batch, **metric_mb.non_tensor_batch, "pad_token_id": pad_token_id}
                    metric_outputs = self._forward_micro_batch(
                        metric_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        disable_inplace_backward=True,
                    )
                    metric_log_prob = metric_outputs["log_probs"]
                    metric_old_lp = metric_log_prob.detach() if use_self_baseline else metric_inputs["old_log_probs"]
                    metric_rmask = metric_inputs["response_mask"]

                    log_ratio = (metric_log_prob - metric_old_lp).clamp(min=-20, max=20)
                    ratio_mse = torch.square(torch.exp(log_ratio) - 1.0)

                    mse_sum += (ratio_mse * metric_rmask).sum().item()
                    mse_count += metric_rmask.sum().item()

                    if entropy_threshold is not None and "rollout_entropy" in metric_inputs:
                        metric_ent = metric_inputs["rollout_entropy"]
                        low_mask = (metric_ent <= entropy_threshold).float() * metric_rmask
                        high_mask = (metric_ent > entropy_threshold).float() * metric_rmask
                        mse_low_sum += (ratio_mse * low_mask).sum().item()
                        mse_low_count += low_mask.sum().item()
                        mse_high_sum += (ratio_mse * high_mask).sum().item()
                        mse_high_count += high_mask.sum().item()

                if was_training:
                    self.actor_module.train()
                else:
                    self.actor_module.eval()

                return {
                    "all": (mse_sum / mse_count) if mse_count > 0 else None,
                    "low": (mse_low_sum / mse_low_count) if mse_low_count > 0 else None,
                    "high": (mse_high_sum / mse_high_count) if mse_high_count > 0 else None,
                }

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        debug_step0_kl = os.getenv("VERL_DEBUG_KL_STEP0", "0") == "1"
        step0_self_baseline = os.getenv("VERL_STEP0_SELF_BASELINE", "1") == "1"
        global_step_idx = 0
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # Scheduled ECA: per-step softmax reweighting of advantages
                if eca_softmax_gamma_schedule is not None and "grad_sq" in mini_batch.batch.keys():
                    gamma = eca_softmax_gamma_schedule[global_step_idx]
                    response_mask = mini_batch.batch["response_mask"]
                    grad_sq = mini_batch.batch["grad_sq"] * response_mask
                    T = response_mask.sum(dim=-1, keepdim=True)
                    logits = gamma * grad_sq
                    logits = logits.masked_fill(response_mask == 0, float("-inf"))
                    weights = torch.softmax(logits, dim=-1)
                    mini_batch.batch["advantages"] = mini_batch.batch["advantages"] * T * weights
                global_step_idx += 1

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                step_idx = global_step_idx - 1  # 0-indexed (incremented above)

                current_entropy_threshold = _compute_entropy_threshold(mini_batch)

                # Pre-step KL metrics (all computed before gradient update)
                use_step0_self_baseline = step0_self_baseline and step_idx == 0
                first_mb_metrics = _compute_prestep_k3_metrics(
                    target_mini_batch=first_metric_mini_batch,
                    entropy_threshold=first_entropy_threshold,
                    mode="eval",
                    use_self_baseline=use_step0_self_baseline,
                )
                last_mb_metrics = _compute_prestep_k3_metrics(
                    target_mini_batch=last_metric_mini_batch,
                    entropy_threshold=last_entropy_threshold,
                    mode="eval",
                    use_self_baseline=use_step0_self_baseline,
                )
                current_mb_train_metrics = _compute_prestep_k3_metrics(
                    target_mini_batch=mini_batch,
                    entropy_threshold=current_entropy_threshold,
                    mode="train",
                    use_self_baseline=use_step0_self_baseline,
                )
                current_mb_eval_metrics = _compute_prestep_k3_metrics(
                    target_mini_batch=mini_batch,
                    entropy_threshold=current_entropy_threshold,
                    mode="eval",
                    use_self_baseline=use_step0_self_baseline,
                )
                current_mb_train_is_ratio_mse_metrics = _compute_prestep_is_ratio_mse_metrics(
                    target_mini_batch=mini_batch,
                    entropy_threshold=current_entropy_threshold,
                    mode="train",
                    use_self_baseline=use_step0_self_baseline,
                )

                metric_groups = {
                    "actor/kl_k3_first_minibatch_estimate": first_mb_metrics,
                    "actor/kl_k3_last_minibatch_estimate": last_mb_metrics,
                    "actor/kl_k3_current_minibatch_train": current_mb_train_metrics,
                    "actor/kl_k3_current_minibatch_eval": current_mb_eval_metrics,
                    # Explicit name for the standard PPO ratio mode:
                    # ref = old_log_probs (computed in eval mode), theta = current log_prob (train mode).
                    "actor/kl_k3_current_minibatch_standard_training_ratio": current_mb_train_metrics,
                    # Standard IS-ratio off-policyness: (exp(log_prob - old_log_prob) - 1)^2
                    "actor/is_ratio_mse_current_minibatch_standard_training_ratio": current_mb_train_is_ratio_mse_metrics,
                }
                for metric_prefix, metric_values in metric_groups.items():
                    if metric_values["all"] is not None:
                        metrics[f"{metric_prefix}_step_{step_idx}"] = metric_values["all"]
                    if metric_values["low"] is not None:
                        metrics[f"{metric_prefix}_low_entropy_step_{step_idx}"] = metric_values["low"]
                    if metric_values["high"] is not None:
                        metrics[f"{metric_prefix}_high_entropy_step_{step_idx}"] = metric_values["high"]

                if debug_step0_kl and step_idx == 0:
                    with torch.no_grad():
                        was_training = self.actor_module.training
                        self.actor_module.eval()

                        debug_micro_batches = _split_metric_micro_batches(mini_batch)

                        abs_diff_old_new_sum = 0.0
                        abs_diff_old_new_count = 0.0
                        abs_diff_new_new_sum = 0.0
                        abs_diff_new_new_count = 0.0
                        max_abs_diff_old_new = 0.0
                        max_abs_diff_new_new = 0.0
                        nonfinite_old = 0.0
                        nonfinite_new = 0.0
                        k3_old_new_weighted = 0.0
                        k3_new_new_weighted = 0.0
                        mse_old_new_weighted = 0.0
                        mse_new_new_weighted = 0.0

                        for debug_mb in debug_micro_batches:
                            debug_mb = debug_mb.to(get_device_id())
                            debug_inputs = {**debug_mb.batch, **debug_mb.non_tensor_batch, "pad_token_id": pad_token_id}
                            out1 = self._forward_micro_batch(
                                debug_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                disable_inplace_backward=True,
                            )
                            out2 = self._forward_micro_batch(
                                debug_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                disable_inplace_backward=True,
                            )

                            old_lp = debug_inputs["old_log_probs"]
                            lp1 = out1["log_probs"]
                            lp2 = out2["log_probs"]
                            rmask = debug_inputs["response_mask"]

                            diff_old_new = lp1 - old_lp
                            diff_new_new = lp2 - lp1
                            abs_old_new = diff_old_new.abs()
                            abs_new_new = diff_new_new.abs()

                            abs_diff_old_new_sum += (abs_old_new * rmask).sum().item()
                            abs_diff_old_new_count += rmask.sum().item()
                            abs_diff_new_new_sum += (abs_new_new * rmask).sum().item()
                            abs_diff_new_new_count += rmask.sum().item()

                            max_abs_diff_old_new = max(max_abs_diff_old_new, float(abs_old_new.max().item()))
                            max_abs_diff_new_new = max(max_abs_diff_new_new, float(abs_new_new.max().item()))

                            finite_old = torch.isfinite(old_lp).float()
                            finite_new = torch.isfinite(lp1).float()
                            nonfinite_old += ((1.0 - finite_old) * rmask).sum().item()
                            nonfinite_new += ((1.0 - finite_new) * rmask).sum().item()

                            k3_old_new = kl_penalty(logprob=old_lp, ref_logprob=lp1, kl_penalty="k3")
                            k3_new_new = kl_penalty(logprob=lp1, ref_logprob=lp2, kl_penalty="k3")
                            k3_old_new_weighted += (k3_old_new * rmask).sum().item()
                            k3_new_new_weighted += (k3_new_new * rmask).sum().item()

                            log_ratio_old_new = (lp1 - old_lp).clamp(min=-20, max=20)
                            log_ratio_new_new = (lp2 - lp1).clamp(min=-20, max=20)
                            mse_old_new = torch.square(torch.exp(log_ratio_old_new) - 1.0)
                            mse_new_new = torch.square(torch.exp(log_ratio_new_new) - 1.0)
                            mse_old_new_weighted += (mse_old_new * rmask).sum().item()
                            mse_new_new_weighted += (mse_new_new * rmask).sum().item()

                        total_tokens = max(abs_diff_old_new_count, 1.0)
                        total_tokens2 = max(abs_diff_new_new_count, 1.0)
                        metrics["debug/step0_abs_logprob_diff_old_vs_new_mean"] = abs_diff_old_new_sum / total_tokens
                        metrics["debug/step0_abs_logprob_diff_new_vs_new_mean"] = abs_diff_new_new_sum / total_tokens2
                        metrics["debug/step0_abs_logprob_diff_old_vs_new_max"] = max_abs_diff_old_new
                        metrics["debug/step0_abs_logprob_diff_new_vs_new_max"] = max_abs_diff_new_new
                        metrics["debug/step0_nonfinite_old_logprobs_tokens"] = nonfinite_old
                        metrics["debug/step0_nonfinite_recomputed_logprobs_tokens"] = nonfinite_new
                        metrics["debug/step0_k3_old_vs_new"] = k3_old_new_weighted / total_tokens
                        metrics["debug/step0_k3_new_vs_new"] = k3_new_new_weighted / total_tokens2
                        metrics["debug/step0_is_ratio_mse_old_vs_new"] = mse_old_new_weighted / total_tokens
                        metrics["debug/step0_is_ratio_mse_new_vs_new"] = mse_new_new_weighted / total_tokens2

                        if was_training:
                            self.actor_module.train()
                        else:
                            self.actor_module.eval()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0) or entropy_top

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # On-policy ECA softmax: reweight advantages using fresh grad_sq from current policy
                    if eca_softmax_on_policy_grad_sq and "sum_pi_squared" in outputs:
                        sum_pi_squared = outputs["sum_pi_squared"].detach()
                        grad_sq_fresh = (1.0 - sum_pi_squared).clamp(min=0.0)
                        T = response_mask.sum(dim=-1, keepdim=True)
                        eca_logits = eca_softmax_gamma * grad_sq_fresh * response_mask
                        eca_logits = eca_logits.masked_fill(response_mask == 0, float("-inf"))
                        weights = torch.softmax(eca_logits, dim=-1)
                        advantages = advantages * T * weights

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # Entropy-top: mask loss to only top X% highest-entropy tokens
                    if entropy_top and entropy is not None:
                        from verl.trainer.ppo.core_algos import get_global_entropy_top_mask

                        entropy_top_mask = get_global_entropy_top_mask(
                            entropy=entropy, response_mask=response_mask, top_ratio=entropy_top_ratio
                        )
                        response_mask = response_mask * entropy_top_mask

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
