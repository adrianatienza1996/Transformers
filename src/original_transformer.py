from wsgiref import handlers
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


#########################################################################################
################################ BASIC BLOCKS ###########################################
#########################################################################################

class MultiHead_Attention(nn.Module):

    def __init__(self, model_dim=512, number_heads=8, do_prob=0.1):
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


    def forward(self, q, k, v, mask):
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
    def __init__(self, model_dim=512, wide_factor=4, do_prob=0.1):
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
    
    def __init__(self, model_dim=512):
        super(Add_and_Norm, self).__init__()
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, res):
        return self.norm(x + res)


#########################################################################################
##################################### ENCODER ###########################################
#########################################################################################

class EncoderBlock(nn.Module):
    def __init__(self, number_heads=8, model_dim=512, do_prob=0.1):
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
    def __init__(self, num_blocks, num_heads=8, model_dim=512, do_prob=0.1):
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



#########################################################################################
##################################### DECODER ###########################################
#########################################################################################

class DecoderBlock(nn.Module):

    def __init__(self, number_heads=8, model_dim=512, do_prob=0.1):
        super(DecoderBlock, self).__init__()
        self.masked_mh_atten = MultiHead_Attention(number_heads=number_heads, 
                                                    model_dim=model_dim,
                                                    do_prob=do_prob)
        self.add_norm_mmh = Add_and_Norm(model_dim=model_dim)
        self.mh_atten = MultiHead_Attention(number_heads=number_heads, 
                                                  model_dim=model_dim,
                                                  do_prob=do_prob)
        self.add_norm_mh = Add_and_Norm(model_dim=model_dim)
        self.ffn = FeedForwardNet(model_dim=model_dim, 
                                  do_prob=do_prob)
        self.add_norm_ffn = Add_and_Norm(model_dim=model_dim)

    
    def forward(self, x, end_rep, mask):
        res = x
        h = self.masked_mh_atten(x, x, x, mask)
        h = self.add_norm_mmh(h, res)
        
        res = h
        h = self.mh_atten(end_rep, end_rep, h)
        h = self.add_norm_mmh(h, res)

        res = h
        h = self.ffn(h)
        return self.add_norm_ffn(h, res)



class Decoder(nn.Module):
    def __init__(self, num_blocks, number_heads=8, model_dim=512, do_prob=0.1):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        self.num_heads = number_heads
        self.model_dim = model_dim
        self.do_prob = do_prob

        self.net = self.create_net()

    def forward(self, x, enc_rep, mask):
        h = x
        for layer in self.net:
            h = layer(h, enc_rep, mask)

        return h

    def create_net(self):
        net = nn.ModuleList()

        for _ in range(self.num_blocks):
            net.append(DecoderBlock(
                number_heads=self.num_heads,
                model_dim=self.model_dim,
                do_prob=self.do_prob))

        return net

