import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Dense, Conv2D, AveragePooling2D
from tensorflow_addons.layers import GroupNormalization
import numpy as np

'''
code adopted from https://github.com/hojonathanho/diffusion and https://github.com/ermongroup/ddim 
the blocks here are meant to be used in a DDPM_checkpoint_Model or DDIM_checkpoint_Model. 
was_pytorch argument for resnet_block and attn_block:
depending on the ordering of the variables in the checkpoints, layers must be initialized in different order to match the initial order.
as a result, the celeba model that was loaded from a pytorch file defines layers differently. Call method is kept the same.
'''
def get_timestep_embedding(timesteps, embedding_dim: int):
    assert len(timesteps.shape) == 1 

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb

class downsample(Layer):
    def __init__(self, c, with_conv):
        super().__init__()
        if with_conv:
            self.down = Conv2D(c, 3, padding='same', strides=2)
        else:
            self.down = AveragePooling2D()
    
    def call(self, x, index):
        return self.down(x)
      
class upsample(Layer):
    def __init__(self, c, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.up = Conv2D(c, 3, padding='same')

    def call(self, x, index):
        B, H, W, C = x.shape
        x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.with_conv:
            x = self.up(x)
        return x

class resnet_block(Layer):
    def __init__(self, c, was_pytorch, use_nin_shortcut=False, drop_rate=0.0):
        super().__init__() 
        self.c = c
        self.drop_rate = drop_rate
        if was_pytorch:
            self.norm1 = GroupNormalization(groups=32)
            self.conv1 = Conv2D(c, 3, padding='same')
            self.temb_proj = Dense(c)
            self.norm2 = GroupNormalization(groups=32)
            self.conv2 = Conv2D(c, 3, padding='same')
            if use_nin_shortcut:
                self.skip_conv = Dense(c)
            else:
                self.skip_conv = None
        else:
            self.conv1 = Conv2D(c, 3, padding='same')
            self.conv2 = Conv2D(c, 3, padding='same')
            if use_nin_shortcut:
                self.skip_conv = Dense(c)
            else:
                self.skip_conv = None

            self.norm1 = GroupNormalization(groups=32) 
            self.norm2 = GroupNormalization(groups=32)
            self.temb_proj = Dense(c) 

    def call(self, x, index):
        residual = tf.identity(x)
        x = tf.nn.swish(self.norm1(x))
        x = self.conv1(x)
        
        x += self.temb_proj(tf.nn.swish(index))[:, None, None, :]
        x = tf.nn.swish(self.norm2(x))
        x = self.conv2(x)

        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        
        return x + residual      

class attn_block(Layer):
    def __init__(self, c, was_pytorch):
        super().__init__() 
        self.c = c
        if was_pytorch:
            self.norm = GroupNormalization(groups=32)
            self.q = Dense(c)
            self.k = Dense(c)
            self.v = Dense(c)
            self.proj_out = Dense(c)
        else:
            self.k = Dense(c)
            self.norm = GroupNormalization(groups=32)
            self.proj_out = Dense(c)
            self.q = Dense(c)
            self.v = Dense(c)
        
    def call(self, x, index):
        B, H, W, C = x.shape
        residual = tf.identity(x)
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
        w = tf.reshape(w, [B, H, W, H * W])
        w = tf.nn.softmax(w, -1)
        w = tf.reshape(w, [B, H, W, H, W])
        x = tf.einsum('bhwHW,bHWc->bhwc', w, v)

        x = self.proj_out(x)
        return x + residual