import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange

from einops import repeat


# The iniput size will be (224, 224)
# We will separate the input in (16, 16) patch size

class Patch_Embeddings(nn.Module):
    def __init__(self, patch_size, model_dim, in_channels, conv_projection):
        super(Patch_Embeddings, self).__init__()
        # According papers, projector can be both Linear Layer or Conv Layer
        self.conv_projection = conv_projection

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b d p1 p2 -> b (p1 p2) d')) if self.conv_projection == True \
        \
        else nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size[0], s2=patch_size[1]),
            nn.Linear(patch_size[0] * patch_size[1] * in_channels, model_dim))

        self.class_token = nn.Parameter(torch.randn(1, 1, model_dim))

    def forward(self, x):

        h = self.net(x)
        
        # Cloning the class embedding token for every object in the batch (p = # patches, d = model dimension)
        cls_token = repeat(self.class_token, '() p d -> b p d', b=x.shape[0])
        h = torch.cat([cls_token, h], dim=1)
        return h


class PositionalEmbedding(nn.Module):
    def __init__(self, model_dim, do_prob, max_seq_len=5000):

        super(PositionalEmbedding, self).__init__()
        self.model_dim = model_dim

        position_id = torch.arange(0, max_seq_len).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dim, 2, dtype=torch.float) / model_dim)
        positional_encodings = torch.zeros(max_seq_len, model_dim)
        positional_encodings[:, 0::2] = torch.sin(position_id * frequencies)  
        positional_encodings[:, 1::2] = torch.cos(position_id * frequencies)  
        self.register_buffer('positional_encodings', positional_encodings)

        self.dropout = nn.Dropout(do_prob)
        

    def forward(self, x):
        tmp_encoding = self.positional_encodings[:x.shape[1]]
        return self.dropout(x + tmp_encoding)



class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, model_dim):
        super(LearnablePositionalEmbedding, self).__init__()
        self.positional_encodings = nn.Parameter(torch.randn((input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1]) + 1, model_dim))

    def forward(self, x):
        return x + self.positional_encodings



class MultiHead_Attention(nn.Module):

    def __init__(self, model_dim, number_heads, do_prob):
        super(MultiHead_Attention, self).__init__()
        self.number_heads = number_heads

        self.scale_factor = 1 / ((model_dim / number_heads) ** 0.5)
        self.att_drop_out = nn.Dropout(do_prob)
        self.output_drop_out = nn.Dropout(do_prob)

        self.block_output = nn.Linear(model_dim, model_dim)

        self.split_head = Rearrange('b l (h d) -> b h l d', h = self.number_heads)
        self.split_head_t = Rearrange('b l (h d) -> b h d l', h = self.number_heads)
        self.concat = Rearrange('b h l d -> b l (h d)') 

        self.x_to_q = nn.Linear(model_dim, model_dim)
        self.x_to_k = nn.Linear(model_dim, model_dim)
        self.x_to_v = nn.Linear(model_dim, model_dim)


    def forward(self, q, k, v, mask=None):
        # q, k and v with shape (batch_size, seq_len, embedding_dimension)
        q = self.split_head(self.x_to_q(q))
        k_transpose = self.split_head_t(self.x_to_k(k))
        v = self.split_head(self.x_to_v(v))

        attention = torch.matmul(q, k_transpose)
        attention = attention * self.scale_factor
        if mask is not None:
            attention.masked_fill_(mask == torch.tensor(False), float("-inf"))
        
        attention = self.att_drop_out(attention.softmax(-1))
        output = torch.matmul(attention, v)
        output = self.concat(output)
        output = self.block_output(output)
        return self.output_drop_out(output)



class FeedForwardNet(nn.Module):
    def __init__(self, model_dim, do_prob, wide_factor=4):
        super(FeedForwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim * wide_factor),
            nn.GELU(),
            nn.Dropout(do_prob),
            nn.Linear(model_dim * wide_factor, model_dim),
            nn.Dropout(do_prob)
        )

    def forward(self, x):
        return self.net(x)



class Add_and_Norm(nn.Module):
    
    def __init__(self, model_dim):
        super(Add_and_Norm, self).__init__()
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, res):
        return self.norm(x + res)



class EncoderBlock(nn.Module):
    def __init__(self, number_heads, model_dim, do_prob):
        super().__init__()
        self.mh_atten_block = MultiHead_Attention(number_heads=number_heads, 
                                                  model_dim=model_dim,
                                                  do_prob=do_prob)
        
        self.add_norm_mh = Add_and_Norm(model_dim=model_dim)
        self.ffn = FeedForwardNet(model_dim=model_dim, 
                                  do_prob=do_prob)

        self.add_norm_ffn = Add_and_Norm(model_dim=model_dim)

    def forward(self, x):
        res = x
        h = self.mh_atten_block(x, x, x)
        h = self.add_norm_mh(h, res)
        
        res = h
        h = self.ffn(h)
        return self.add_norm_ffn(h, res)


class Encoder(nn.Module):
    def __init__(self, num_blocks, num_heads, model_dim, do_prob):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.num_heads  = num_heads
        self.model_dim = model_dim
        self.do_prob = do_prob

        self.net = self.create_net()

    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h)

        return h

    def create_net(self):
        net = nn.ModuleList()

        for _ in range(self.num_blocks):
            net.append(EncoderBlock(
                number_heads=self.num_heads,
                model_dim=self.model_dim,
                do_prob=self.do_prob))

        return net


class ViT(nn.Module):
    def __init__(self, num_classes, num_blocks=6, num_heads=8, model_dim=512, do_prob=0.1, patch_size=(16, 16), input_size=(224, 224, 3), conv_projection=True, learnable_pos=True):
        super(ViT, self).__init__()
        self.patch_embedding = Patch_Embeddings(patch_size=patch_size, model_dim=model_dim, in_channels=input_size[2], conv_projection=conv_projection)
        
        self.pos_encoding = LearnablePositionalEmbedding(input_size=(input_size[0], input_size[1]), patch_size=patch_size, model_dim=model_dim) if learnable_pos\
            else PositionalEmbedding(model_dim=model_dim)

        self.encoder = Encoder(num_blocks=num_blocks, num_heads=num_heads, model_dim=model_dim, do_prob=do_prob)
        self.mlp = nn.Linear(model_dim, num_classes)


    def forward(self, x):
        h = self.patch_embedding(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h = h[:, 0, :]
        return self.mlp(h)



