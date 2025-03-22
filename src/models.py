import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class NewGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gaussian Error Linear Unit (GELU) activation function.

        Args:
            input: a tensor containing the input data.

        Returns:
            A tensor where the GELU activation has been applied element-wise.
        """
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * input**3))
            )
        )


class PatchEmbeddings(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the PatchEmbeddings module.

        Args:
            config: a dictionary containing configuration parameters:
                - "image_size": the size (height/width) of the input image.
                - "patch_size": the size (height/width) of each patch.
                - "num_channels": number of channels in the input image.
                - "hidden_size": the dimension to project each patch to.
        """
        super().__init__()
        self.image_size: int = config["image_size"]
        self.patch_size: int = config["patch_size"]
        self.num_channels: int = config["num_channels"]
        self.hidden_size: int = config["hidden_size"]

        # calculate the total number of patches per image
        self.num_patches: int = (self.image_size // self.patch_size) ** 2

        # convolution to project patches
        self.projection: nn.Conv2d = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert an image into patch embeddings.

        Args:
            x: input tensor with shape (batch_size, num_channels, image_size, image_size).

        Returns:
            A tensor with shape (batch_size, num_patches, hidden_size) where each row
            corresponds to a flattened and projected patch embedding.
        """
        # apply convolution to split the image into patches and project them
        x = self.projection(x)

        # flatten the patches
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the Embeddings module which combines patch embeddings with
        a class token and positional embeddings.

        Args:
            config: a dictionary with the following keys:
                - "hidden_size": dimension of the patch embeddings.
                - "hidden_dropout_prob": dropout probability.
                - Plus keys required by PatchEmbeddings (e.g., "image_size", "patch_size", "num_channels").
        """
        super().__init__()

        # generate patch embeddings from the input image
        self.patch_embeddings: nn.Module = PatchEmbeddings(config)

        # create a learnable [CLS] token for classification
        self.cls_token: nn.Parameter = nn.Parameter(
            torch.randn(1, 1, config["hidden_size"])
        )

        # create learnable positional embeddings for the patches and the [CLS] token
        self.position_embeddings: nn.Parameter = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )

        # dropout layer for regularization.
        self.dropout: nn.Dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Embeddings module.

        Args:
            x: Input tensor of shape (batch_size, num_channels, image_size, image_size).

        Returns:
            Tensor of shape (batch_size, num_patches + 1, hidden_size) with combined patch embeddings,
            class token, and positional embeddings.
        """
        # obtain patch embeddings from the input image
        x = self.patch_embeddings(x)
        batch_size: int = x.shape[0]

        # expand the [CLS] token to the batch size and concatenate the [CLS] token at the beginning of the patch embeddings
        cls_tokens: torch.Tensor = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add the positional embeddings and apply dropout
        x = x + self.position_embeddings
        x = self.dropout(x)

        return x


class AttentionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_head_size: int,
        dropout: float,
        bias: bool = True,
    ):
        """
        Initializes an Attention Head module.

        Args:
            hidden_size: dimension of the input embeddings.
            attention_head_size: dimension of the projection for this attention head.
            dropout: dropout probability applied to the attention probabilities.
            bias: whether to include bias terms in the linear projections.
        """
        super().__init__()
        # linear layers for computing query, key, and value vectors
        self.query: nn.Linear = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key: nn.Linear = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value: nn.Linear = nn.Linear(hidden_size, attention_head_size, bias=bias)

        # dropout applied to the attention probabilities
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.attention_head_size: int = attention_head_size  # save it for scaling

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the attention head.

        Args:
            x: input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
            A tuple containing:
                - attn_output: tensor of shape (batch_size, seq_length, attention_head_size)
                resulting from applying the attention mechanism.
                - attn_probs: tensor of shape (batch_size, seq_length, seq_length)
                containing the attention probabilities.
        """
        # project input to query, key, and value spaces
        query: torch.Tensor = self.query(x)
        key: torch.Tensor = self.key(x)
        value: torch.Tensor = self.value(x)

        # compute scaled dot-product attention scores
        attn_scores: torch.Tensor = torch.matmul(
            query, key.transpose(-1, -2)
        ) / math.sqrt(self.attention_head_size)

        # normalize scores with softmax and apply dropout to the attention probabilities
        attn_probs: torch.Tensor = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # compute the attention output as the weighted sum of the values
        attn_output: torch.Tensor = torch.matmul(attn_probs, value)

        return attn_output, attn_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the Multi-Head Attention module.

        Args:
            config: a dictionary containing configuration parameters:
                - "num_attention_heads": number of attention heads.
                - "hidden_size": input and output feature dimension.
                - "qkv_bias": whether to include bias in the linear projections.
                - "attention_probs_dropout_prob": dropout probability for attention probabilities.
                - "hidden_dropout_prob": dropout probability for the output projection.
        """
        super().__init__()
        self.num_attention_heads: int = config["num_attention_heads"]
        self.hidden_size: int = config["hidden_size"]

        # compute the size of each attention head
        self.attention_head_size: int = self.hidden_size // self.num_attention_heads

        # total size when concatenating all heads
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size
        self.qkv_bias: bool = config["qkv_bias"]

        # list of individual attention heads
        self.heads: nn.ModuleList = nn.ModuleList(
            [
                AttentionHead(
                    self.hidden_size,
                    self.attention_head_size,
                    config["attention_probs_dropout_prob"],
                    self.qkv_bias,
                )
                for _ in range(self.num_attention_heads)
            ]
        )

        # apply a linear projection
        self.output_projection: nn.Linear = nn.Linear(
            self.all_head_size, self.hidden_size
        )
        self.output_dropout: nn.Dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Multi-Head Attention module.

        Args:
            x: input tensor of shape (batch_size, seq_length, hidden_size).
            output_attentions: if True, also return attention probabilities for each head.

        Returns:
            A tuple containing:
                - attention_output: tensor of shape (batch_size, seq_length, hidden_size)
                representing the result of the multi-head attention.
                - Optional attention probabilities: if output_attentions is True,
                a tensor of shape (batch_size, num_heads, seq_length, seq_length) is returned;
                otherwise, None.
        """
        # compute the attention output for each head
        head_outputs = [head(x)[0] for head in self.heads]

        # concatenate and project
        concat_heads: torch.Tensor = torch.cat(head_outputs, dim=-1)
        attention_output: torch.Tensor = self.output_projection(concat_heads)
        attention_output = self.output_dropout(attention_output)

        # gather the attention probabilities from each head
        if output_attentions:
            head_attentions = [head(x)[1] for head in self.heads]
            return attention_output, torch.stack(head_attentions, dim=1)
        else:
            return attention_output, None


class MLP(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the Multi-Layer Perceptron (MLP) module used in the Transformer block.

        Args:
            config: a dictionary containing configuration parameters:
                - "hidden_size": dimension of the input and output.
                - "intermediate_size": dimension of the hidden layer in the MLP.
                - "hidden_dropout_prob": dropout probability applied after the MLP.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(
            config["hidden_size"], config["intermediate_size"]
        )
        self.activation: nn.Module = NewGELUActivation()
        self.fc2: nn.Linear = nn.Linear(
            config["intermediate_size"], config["hidden_size"]
        )
        self.dropout: nn.Dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
            Tensor of the same shape (batch_size, seq_length, hidden_size) after applying
            the MLP transformation, activation, and dropout.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes a single Transformer block which includes multi-head attention and an MLP,
        each preceded by layer normalization and followed by a residual connection.

        Args:
            config: a dictionary containing configuration parameters:
                - "hidden_size": the input and output dimension of the block.
                - Other parameters required by MultiHeadAttention and MLP.
        """
        super().__init__()
        self.attention: nn.Module = MultiHeadAttention(config)
        self.layernorm1: nn.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.mlp: nn.Module = MLP(config)
        self.layernorm2: nn.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-6)

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Transformer block.

        Args:
            x: input tensor of shape (batch_size, seq_length, hidden_size).
            output_attentions: if True, the block also returns the attention probabilities.

        Returns:
            A tuple where:
                - The first element is the output tensor of shape (batch_size, seq_length, hidden_size).
                - The second element is either the attention probabilities tensor
                of shape (batch_size, seq_length, seq_length) or None if output_attentions is False.
        """
        attn_input: torch.Tensor = self.layernorm1(x)
        attn_output, attn_probs = self.attention(attn_input, output_attentions)

        # add the attention output to the input (residual connection)
        x = x + attn_output

        mlp_input: torch.Tensor = self.layernorm2(x)
        mlp_output: torch.Tensor = self.mlp(mlp_input)

        # add the MLP output to the previous result (residual connection)
        x = x + mlp_output

        if output_attentions:
            return x, attn_probs
        else:
            return x, None


class Encoder(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the Transformer encoder module by stacking multiple Transformer blocks.

        Args:
            config: a dictionary containing configuration parameters, including:
                - "num_hidden_layers": number of Transformer blocks to stack.
                - Other parameters required by the Block module.
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block(config) for _ in range(config["num_hidden_layers"])]
        )

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through the encoder.

        Args:
            x: input tensor of shape (batch_size, seq_length, hidden_size).
            output_attentions: if True, collects and returns the attention probabilities from each block.

        Returns:
            A tuple containing:
                - The final encoder output tensor of shape (batch_size, seq_length, hidden_size).
                - A list of attention probability tensors from each block if output_attentions is True;
                otherwise, None.
        """
        all_attentions: List[torch.Tensor] = []
        # process the input through each block
        for block in self.blocks:
            x, attn_probs = block(x, output_attentions)
            if output_attentions:
                if attn_probs is None:
                    attn_probs = torch.empty(0)  # valid tensor appended
                all_attentions.append(attn_probs)
        if output_attentions:
            return x, all_attentions
        return x, None


class ViTForClassification(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the Vision Transformer (ViT) model for image classification.

        Args:
            config: a dictionary containing configuration parameters, including:
                - "hidden_size": dimension of the patch embeddings.
                - "num_classes": number of target classes for classification.
                - "initializer_range": standard deviation for weight initialization.
                - Additional keys required by the Embeddings and Encoder modules.
        """
        super().__init__()
        self.config = config
        self.embedding: nn.Module = Embeddings(config)
        self.encoder: nn.Module = Encoder(config)
        self.classifier: nn.Linear = nn.Linear(
            config["hidden_size"], config["num_classes"]
        )
        self.apply(self._init_weights)

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for the ViT model.

        Args:
            x: input tensor of shape (batch_size, num_channels, image_size, image_size).
            output_attentions: if True, returns attention probabilities from the encoder.

        Returns:
            A tuple where:
                - The first element is the logits tensor of shape (batch_size, num_classes).
                - The second element is either a list of attention probability tensors (if output_attentions is True)
                or None.
        """
        embeddings: torch.Tensor = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embeddings, output_attentions)

        # output corresponding to the [CLS] token for classification.
        logits: torch.Tensor = self.classifier(encoder_output[:, 0, :])

        if output_attentions:
            return logits, all_attentions
        return logits, None

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes weights for linear and convolutional layers using truncated normal distribution,
        and initializes biases to zero. Also initializes layer norm weights and the embedding parameters.

        Args:
            module: the module to initialize.
        """

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(
                module.weight,
                std=self.config["initializer_range"],
                a=-2 * self.config["initializer_range"],
                b=2 * self.config["initializer_range"],
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, Embeddings):
            nn.init.trunc_normal_(
                module.position_embeddings, std=self.config["initializer_range"]
            )
            nn.init.trunc_normal_(
                module.cls_token, std=self.config["initializer_range"]
            )
