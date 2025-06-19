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
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from tensordict import TensorDict
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_scvit import SaccadicViTConfig
from .predictor import (
    SACCADIC_VIT_PREDICTOR_CLASSES,
    AbstractSaccadicViTPredictor,
)


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


PATCH_CONFIG_DOF: Dict[str, int] = {
    "translation": 2,
    # "euclidean": 3,
    "scaling": 3,
    # "similarity": 4,
    "non-uniform-scaling": 4,
    # "non-shear": 5,
    # "affine": 6,
}


class SaccadicViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()

        self.patch_embeddings = SaccadicViTPatchEmbeddings(config)
        self.position_encoder = nn.Linear(PATCH_CONFIG_DOF[config.patch_config], config.hidden_size, bias=config.pe_bias)
        self.position_decoder = nn.Linear(config.hidden_size, PATCH_CONFIG_DOF[config.patch_config], bias=config.pe_bias)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def sample_initial(self) -> torch.Tensor:
        match self.config.patch_config:
            case "translation" | "scaling" | "non-uniform-scaling":
                return torch.zeros((PATCH_CONFIG_DOF[self.config.patch_config],))

            case _:
                raise ValueError(self.config.patch_config)

    def forward(
        self,
        pixel_values: torch.Tensor,     # float: [B... x C x H x W]
        patch_config: torch.Tensor,     # float: [B... x ?]
    ) -> torch.Tensor:                  # float: [B... x D]
        patch_embeddings = self.patch_embeddings(pixel_values, patch_config)
        positional_embeddings = self.position_encoder(patch_config)

        # add positional encoding to each token
        embeddings = patch_embeddings + positional_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class SaccadicViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: SaccadicViTConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        self.image_size = image_size
        self.patch_config = config.patch_config

        self.grid: torch.Tensor = torch.stack(torch.meshgrid(
            torch.linspace(-1.0, 1.0, patch_size),
            torch.linspace(-1.0, 1.0, patch_size),
        ) + (torch.ones((patch_size, patch_size)),), dim=-1)   # float: [P x P x 3]

        num_channels, hidden_size = config.num_channels, config.hidden_size
        self.num_channels = num_channels
        self.projection = nn.Sequential(
            nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size),
            nn.Flatten(),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,     # float: [B... x C x H x W]
        patch_config: torch.Tensor,     # float: [B... x ?]
    ) -> torch.Tensor:                  # float: [B... x D]
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        t = patch_config[..., :2]
        match self.patch_config:
            case "translation":
                I = torch.eye(2).expand(patch_config.shape + (2,))      # float: [B... x 2 x 2]
                t_ = patch_config[..., None, :]                         # float: [B... x 1 x 2]
                affine_transform = torch.cat((I, t_), dim=-2)           # float: [B... x 3 x 2]

            case "scaling":
                I = torch.eye(2).expand(patch_config.shape + (2,))      # float: [B... x 2 x 2]
                D = I * torch.exp(patch_config[..., 2, None, None])
                t_ = patch_config[..., None, :]                         # float: [B... x 1 x 2]
                affine_transform = torch.cat((D, t_), dim=-2)           # float: [B... x 3 x 2]

            # case "similarity":
            #     t = patch_config[..., :2]                               # float: [B... x 2]
            #     u = patch_config[..., 2:]                               # float: [B... x 2]
            #     v = torch.stack((-u[..., 1], u[..., 0]), dim=-1)        # float: [B... x 2]
            #     affine_transform = torch.stack((u, v, t), dim=-2)       # float: [B... x 3 x 2]

            case "non_uniform_scaling":
                D = torch.diag_embed(torch.exp(patch_config[..., 2:4])) # float: [B... x 2 x 2]
                t_ = patch_config[..., None, :]                         # float: [B... x 1 x 2]
                affine_transform = torch.cat((D, t_), dim=-2)           # float: [B... x 3 x 2]

            case _:
                raise ValueError(self.patch_config)

        sample_grid = self.grid @ affine_transform[..., None, :, :]     # float: [B... x P x P x 2]
        sample_patches = torch.nn.functional.grid_sample(
            pixel_values, sample_grid.transpose(-3, -2),
            mode="bicubic", padding_mode="zeros",
        )                                                   # float: [B... x C x P x P]
        embeddings = self.projection(sample_patches)        # float: [B... x D]

        return embeddings




















class SaccadicViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SaccadicViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "ViTLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data = nn.init.trunc_normal_(
                module.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.dtype)


VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class SaccadicViTModel(ViTPreTrainedModel):
    def __init__(self, config: SaccadicViTConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = SaccadicViTEmbeddings(config)
        self.predictor: AbstractSaccadicViTPredictor = SACCADIC_VIT_PREDICTOR_CLASSES[config.refiner_implementation](config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> SaccadicViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,

        convergence_distance: Literal["spatial", "latent"] = "spatial",
        absolute_threshold: float = 0.1,
        max_trace_length: int = 100,
        max_saccade_length: int = 10,


        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)





        # TODO: New model logic
        bsz = pixel_values.shape[:-3]
        flattened_pixel_values = pixel_values.view((-1,) + pixel_values.shape[-3:])

        @dataclass
        class Edge:
            pattern_index: int
            node_indices: torch.Tensor

        def supply_empty_history() -> TensorDict[str, torch.Tensor]:
            return TensorDict({
                "predicted_embedding": torch.empty((0, self.config.hidden_size), dtype=torch.get_default_dtype()),
                "query": torch.empty((0, PATCH_CONFIG_DOF[self.config.patch_config]), dtype=torch.get_default_dtype()),
                "embedding": torch.empty((0, self.config.hidden_size), dtype=torch.get_default_dtype()),
                # "parent_index": torch.empty((0,), dtype=torch.long),
            }, batch_size=(0,))

        def min_distance(timestep: TensorDict[str, torch.Tensor], window: TensorDict[str, torch.Tensor]) -> torch.return_types.min:
            key = "query" if convergence_distance == "spatial" else "embedding"
            distances = torch.cdist(timestep[key][None], window[key])[0]
            return torch.min(distances, dim=0)


        encoder_output_list = []
        for pixel_values_ in torch.unbind(flattened_pixel_values, dim=0):

            queue: List[Tuple[Edge, torch.Tensor]] = [(None, None)]                             # list[tuple[int: parent_idx, float: [D]]]
            constructed_edges: List[Edge] = []                                                  # list[tuple[int: pattern_idx, int: [k]]]

            history = supply_empty_history()
            saccade = supply_empty_history()

            terminal_indices: List[int] = []                                                    # list[int]: C x int
            proposed_edges: List[Tuple[Edge, torch.Tensor]] = []                                # list[tuple[tuple[int: pattern_idx, int: [k]], float: [D]]]:

            it = 0
            while it < max_trace_length and len(queue) > 0:
                proposing_edge, predicted_embedding = queue.pop(0)                              # float: [D]
                if predicted_embedding is None:
                    next_query = self.embeddings.sample_initial()                               # float: [dof]
                else:
                    next_query = self.embeddings.position_decoder(predicted_embedding)          # float: [dof]




                # TODO: Use pattern and refiner to determine the refined embedding and potential edges
                encoder_embedding: torch.Tensor = self.embeddings(pixel_values_, next_query)    # float: [D]








                next_embedding: torch.Tensor = None                                             # float: [D]
                proposed_edges = None










                timestep = TensorDict({
                    "predicted_embedding": predicted_embedding,
                    "query": next_query,
                    "embedding": next_embedding,
                }, batch_size=())
                saccade = TensorDict.cat((saccade, timestep[None]), dim=0)






                terminal_index: int = None
                # • If saccade completes, either through convergence or max length, then add query to convergent queries, branch from query, and reset saccade
                if (len(saccade) == max_saccade_length) or (len(saccade) > 0 and min_distance(timestep, saccade[:-1]).values.item() < absolute_threshold):
                    terminal_index = len(history) + len(saccade) - 1

                # • Else if next query is close to historic query, then a loop closure is formed and the pattern proposing the current saccade is imposed on the previous saccade
                elif len(history) > 0 and (history_min_return := min_distance(timestep, history)).values.item() < absolute_threshold:
                    terminal_index = history["terminal_index"][history_min_return.indices].item()


                if terminal_index is not None:
                    saccade["terminal_index"] = torch.full((len(saccade),), terminal_index)
                    history = TensorDict.cat((history, saccade), dim=0)
                    saccade = supply_empty_history()

                    unfilled_mask = (proposing_edge.node_indices == -1)
                    assert torch.sum(unfilled_mask).item() == 1, "Proposing pattern should have exactly one unfilled position."
                    proposing_edge.node_indices[unfilled_mask] = terminal_index

                    if terminal_index not in terminal_indices:
                        terminal_indices.append(terminal_index)
                        queue = proposed_edges + queue

                it += 1








                # # • Else, continue the saccade, computing the true result of the query, and pattern matching to determine potential edges and the next query
                # else:
                #
                #
                #     embedding = self.embeddings.forward(pixel_values_[None], next_query[None])[0]   # float: [D]
                #
                #     embedding_pool = torch.cat((
                #         embedding[None], history["embedding"][convergent_indices]
                #     ), dim=0)                                                                       # float: [(C + 1) x D]
                #
                #     match = self.pattern.match(embedding_pool, output_attentions=False)
                #     relevant_matches: Dict[int, torch.Tensor] = {}
                #     for k, v in match.match_indices.items():                                        # k, int: [P x k]
                #         relevant_mask = torch.any(v == 0, dim=-1)                                   # bool: [P]
                #
                #         pattern = match.match[k][relevant_mask]                                     # int: [P? x (k + 1) x D]
                #         refined_pattern = self.refiner(pattern).hidden_states                       # int: [P? x (k + 1) x D]
                #
                #         preservation_error = torch.norm((pattern - refined_pattern)[:, 1:, :], dim=-1) ** 2     # float: [P? x k]
                #         preservation_error = torch.sum(preservation_error * (v[relevant_mask] != -1), dim=1)    # float: [P?]









        # Batched version
        """
        queue: List[torch.Tensor] = [self.embeddings.sample_initial().expand(bsz + (PATCH_CONFIG_DOF[self.config.patch_config],))]
        convergent_queries: torch.Tensor = torch.zeros(bsz + (0, 2))        # float: [B... x ? x 2]

        previous_query: torch.Tensor = torch.full(bsz + (2,), torch.inf)    # float: [B... x 2]
        saccade_lengths: torch.Tensor = torch.zeros(bsz, dtype=torch.long)
        while len(queue) > 0:
            next_query = queue.pop(0)
        """





        # TODO: Modify this logic
        """"""
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        """"""








        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForMaskedImageModeling(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
