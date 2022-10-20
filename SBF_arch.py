import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 8,
        emb_size: int = 768,
        img_size: int = 224,
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        # Theoretically, the larger the embedding size,
        # the more information can be learnt
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads

        # gets queries by breaking apart each flattened patch.
        # For the default first layer, each image patch is
        # transformed into a new representation using a fully connected
        # layer and then broken into 6 strips. It is not 6
        # horizontal strips as it is transformed under FC
        # Based on your understanding of BP, this should
        # be similar to the idea of group-norm in CNNs in that each head
        # will learn a different representation of each
        # patch, ie effectively this is num_heads FC layers in parallel
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)

        # gets keys and values by doing a similar thing
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # so now we have three different representations (q, k, v) of the original image.

        # Next we want to find the dot product between each query and key.
        # We do this by rearranging (done before) to
        # be batch x head x patches x d-dim rep. Then the dot product
        # between every d-dim rep of every patch for each
        # head is calculated

        # idea! if my understanding of the orthogonality constraint
        # on information flow between representations is
        # correct we should be able to remove either queries or keys
        #  and do the dot product with itself to get similar
        # performance while saving on 1/2 of the compute.

        dotprod = torch.einsum("b h n d, b h m d -> b h n m", queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(dotprod, dim=-1) / scaling
        # soft-max is exponential so dividing by the scaling will
        # make the softmax values smaller so its easier

        # Dropout layer
        att = self.att_drop(att)

        # Perform matrix multiplication to get attention weighted values
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)

        return out


class BudgetAttentionTwo(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        # Theoretically, the larger the embedding size,
        # the more information can be learnt
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # gets keys and values by doing a similar thing
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        dotprod = torch.einsum("b h n d, b h m d -> b h n m", keys, keys)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(dotprod, dim=-1) / scaling
        # soft-max is exponential so dividing by the scaling will
        # make the softmax values smaller so its easier

        # Dropout layer
        att = self.att_drop(att)

        # Perform matrix multiplication to get attention weighted values
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size, emb_size),
        )


class BudgetFeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Dropout(drop_p),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_drop_p: float = 0.0,
        **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class TransformerBudgetEncoderBlockThree(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_drop_p: float = 0.0,
        **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    BudgetAttentionTwo(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    BudgetFeedForwardBlock(emb_size, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerBudgetEncoderThree(nn.Sequential):
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(
            *[TransformerBudgetEncoderBlockThree(**kwargs) for _ in range(depth)]
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )


class ViT(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 8,
        emb_size: int = 128,
        img_size: int = 32,
        depth: int = 12,
        n_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )


class BViTThree(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 8,
        emb_size: int = 128,
        img_size: int = 32,
        depth: int = 12,
        n_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerBudgetEncoderThree(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )
