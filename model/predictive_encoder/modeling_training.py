# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT model."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import einops
import torch
import torch.utils.checkpoint
from torch import nn
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    # replace_return_docstrings,
    # torch_int,
)

from model.predictive_encoder.modeling_spvitencoder import PredictiveViTModel, BaseModelOutputWithInputs, PATCH_CONFIG_DOF
from model.predictive_encoder.configuration_training import PredictiveViTTrainingConfig
from infrastructure.settings import RUNTIME_MODE


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


def compute_error_with_context_lengths(output: torch.Tensor, target: torch.Tensor, context_lengths: torch.Tensor) -> torch.Tensor:
    error = torch.norm(output - target, p="fro", dim=-1) ** 2           # float: [B... x max_N]
    error = sum_error_with_context_lengths(error, context_lengths)      # float: [B...]
    
    return error


def sum_error_with_context_lengths(error: torch.Tensor, context_lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(error.shape[-1]) < context_lengths[..., None]   # bool: [B... x max_N]
    error = torch.sum(error * mask, dim=-1)                             # float: [B...]
    
    return error


def preservation_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    input_context = output.input_hidden_state[..., 1:-1, :] # float: [B... x max_N x D]
    output_context = output.last_hidden_state[..., 1:-1, :] # float: [B... x max_N x D]
    
    error = compute_error_with_context_lengths(
        output_context, input_context, output.context_lengths,
    ) / model.config.expected_context_length                # float: [B...]
    
    return error, {}


# Latent prediction
def prediction_error(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    predicted_state: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    position_config = model.embeddings.position_decoder(predicted_state, return_orthogonal=False)   # float: [B... x N x ?]
    true_state = model.embeddings(pixel_values, position_config)[..., 1:-1, :]  # float: [B... x N x D]
    error = torch.norm(predicted_state - true_state, p="fro", dim=-1) ** 2      # float: [B... x N]
    
    return error, {
        "config": position_config,
        "true_state": true_state,
    }


def context_prediction_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_context_state = output.last_hidden_state[..., 1:-1, :]                                        # float: [B... x max_N x D]
    error, meta = prediction_error(pixel_values, model, predicted_context_state,)    # float: [B... x max_N x ?], [B... x max_N]
    error = sum_error_with_context_lengths(error, output.context_lengths,) / model.config.expected_context_length
     
    return error, {
        "predicted_context_position": meta["config"],
        "true_context_state": meta["true_state"],
    }


def query_prediction_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_query_state = output.last_hidden_state[..., -1:, :]                                       # float: [B... x 1 x D]
    error, meta = prediction_error(pixel_values, model, predicted_query_state,)  # float: [B... x 1 x ?], [B... x 1]

    return error[..., 0], {
        "predicted_query_position": meta["config"][..., 0, :],
        "true_query_state": meta["true_state"][..., 0, :],
    }


# Patch predicition
def patch_prediction_error(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    predicted_state: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    position_config, orthogonal_state = model.embeddings.position_decoder(predicted_state, return_orthogonal=True)  # float: [B... x N x ?]
    
    predicted_patch = model.embeddings.patch_embeddings.decode(orthogonal_state)                    # float: [B... x N x C x P x P]
    true_patch = model.embeddings.patch_embeddings.sample_patches(pixel_values, position_config)    # float: [B... x N x C x P x P]
    
    error = torch.norm(torch.flatten(predicted_patch - true_patch, start_dim=-3, end_dim=-1), p="fro", dim=-1) ** 2 # float: [B... x N]
    
    return error, {
        "config": position_config,
        "predicted_patch": predicted_patch,
        "true_patch": true_patch,
    }


def context_patch_prediction_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_context_state = output.last_hidden_state[..., 1:-1, :]                            # float: [B... x max_N x D]

    error, meta = patch_prediction_error(pixel_values, model, predicted_context_state,)
    error = sum_error_with_context_lengths(error, output.context_lengths,) / (model.config.expected_context_length * model.config.patch_size ** 2)
    
    return error, {
        "predicted_context_position": meta["config"],
        "predicted_context_patch": meta["predicted_patch"],
        "true_context_patch": meta["true_patch"],
    }


def query_patch_prediction_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_query_state = output.last_hidden_state[..., -1:, :]                               # float: [B... x 1 x D]
    error, meta = patch_prediction_error(pixel_values, model, predicted_query_state,)

    return error[..., 0] / (model.config.patch_size ** 2), {
        "predicted_query_position": meta["config"][..., 0, :],
        "predicted_query_patch": meta["predicted_patch"][..., 0, :, :, :],
        "true_query_patch": meta["true_patch"][..., 0, :, :, :],
    }


def positional_recovery_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_context_state = output.last_hidden_state[..., 1:-1, :]                # float: [B... x max_N x D]
    
    input_config = output.input_position                                            # float: [B... x max_N x ?]
    position_config = model.embeddings.position_decoder(predicted_context_state)    # float: [B... x max_N x ?]
    
    error = compute_error_with_context_lengths(
        position_config, input_config, output.context_lengths,
    ) / model.config.expected_context_length
    
    return error, {}


def positional_regularization_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_state = output.last_hidden_state[..., 1:, :]                  # float: [B... x (max_N + 1) x D]
    position_config = model.embeddings.position_decoder(predicted_state)    # float: [B... x (max_N + 1) x ?]
    
    scale = torch.tensor(model.config.patch_config_scale)
    match scale.ndim:
        case 0:
            position_config = position_config / scale
        case 2:
            scale = scale[:PATCH_CONFIG_DOF[model.config.patch_config]]     # float: [? x 2]
            position_config = (position_config - scale[:, 1]) / scale[:, 0]
        case _:
            raise ValueError(scale.ndim)

    error = compute_error_with_context_lengths(
        position_config, 0.0, output.context_lengths,
    ) + torch.norm(position_config[..., -1, :], p="fro", dim=-1) ** 2       # float: [B...]
    error = error / (model.config.expected_context_length + 1)
    
    return error, {}


LOSS2FN: Dict[str, Callable[[torch.Tensor, PredictiveViTModel, BaseModelOutputWithInputs], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]] = {
    "preservation": preservation_loss,
    "context_prediction": context_prediction_loss,
    "query_prediction": query_prediction_loss,
    "context_patch_prediction": context_patch_prediction_loss,
    "query_patch_prediction": query_patch_prediction_loss,
    "positional_recovery": positional_recovery_loss,
    "positional_regularization": positional_regularization_loss,
}


def training_loss(
    pixel_values: torch.Tensor,
    model: PredictiveViTModel,
    output: BaseModelOutputWithInputs,
    train_config: PredictiveViTTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses, meta = {}, {}
    error = torch.zeros(pixel_values.shape[:-3])
    for k, v in vars(train_config).items():    
        if v != 0.0 and k in LOSS2FN:
            _error, _meta = LOSS2FN[k](pixel_values, model, output)
            error = error + v * _error
            losses[k] = _error
            meta.update({_k: _v.detach() for _k, _v in _meta.items()})
    return error, losses, meta


    
# class ViTPooler(nn.Module):
#     def __init__(self, config: ViTConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()
#
#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output
#
#
# @add_start_docstrings(
#     """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).
#
#     <Tip>
#
#     Note that we provide a script to pre-train this model on custom data in our [examples
#     directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).
#
#     </Tip>
#     """,
#     VIT_START_DOCSTRING,
# )
# class ViTForMaskedImageModeling(ViTPreTrainedModel):
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__(config)
#
#         self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
#
#         self.decoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=config.hidden_size,
#                 out_channels=config.encoder_stride**2 * config.num_channels,
#                 kernel_size=1,
#             ),
#             nn.PixelShuffle(config.encoder_stride),
#         )
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, MaskedImageModelingOutput]:
#         r"""
#         bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
#             Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
#
#         Returns:
#
#         Examples:
#         ```python
#         >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
#         >>> import torch
#         >>> from PIL import Image
#         >>> import requests
#
#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)
#
#         >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
#         >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")
#
#         >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
#         >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
#         >>> # create random boolean mask of shape (batch_size, num_patches)
#         >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
#
#         >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
#         >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
#         >>> list(reconstructed_pixel_values.shape)
#         [1, 3, 224, 224]
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
#             raise ValueError(
#                 "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
#                 "the reconstructed image has the same dimensions as the input. "
#                 f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
#             )
#
#         outputs = self.vit(
#             pixel_values,
#             bool_masked_pos=bool_masked_pos,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             interpolate_pos_encoding=interpolate_pos_encoding,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         # Reshape to (batch_size, num_channels, height, width)
#         sequence_output = sequence_output[:, 1:]
#         batch_size, sequence_length, num_channels = sequence_output.shape
#         height = width = math.floor(sequence_length**0.5)
#         sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
#
#         # Reconstruct pixel values
#         reconstructed_pixel_values = self.decoder(sequence_output)
#
#         masked_im_loss = None
#         if bool_masked_pos is not None:
#             size = self.config.image_size // self.config.patch_size
#             bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
#             mask = (
#                 bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
#                 .repeat_interleave(self.config.patch_size, 2)
#                 .unsqueeze(1)
#                 .contiguous()
#             )
#             reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
#             masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels
#
#         if not return_dict:
#             output = (reconstructed_pixel_values,) + outputs[1:]
#             return ((masked_im_loss,) + output) if masked_im_loss is not None else output
#
#         return MaskedImageModelingOutput(
#             loss=masked_im_loss,
#             reconstruction=reconstructed_pixel_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#
# @add_start_docstrings(
#     """
#     ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
#     the [CLS] token) e.g. for ImageNet.
#
#     <Tip>
#
#         Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
#         setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
#         position embeddings to the higher resolution.
#
#     </Tip>
#     """,
#     VIT_START_DOCSTRING,
# )
# class ViTForImageClassification(ViTPreTrainedModel):
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__(config)
#
#         self.num_labels = config.num_labels
#         self.vit = ViTModel(config, add_pooling_layer=False)
#
#         # Classifier head
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_IMAGE_CLASS_CHECKPOINT,
#         output_type=ImageClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
#     )
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, ImageClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.vit(
#             pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             interpolate_pos_encoding=interpolate_pos_encoding,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         logits = self.classifier(sequence_output[:, 0, :])
#
#         loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output
#
#         return ImageClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
