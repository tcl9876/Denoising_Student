import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import Model
from tensorflow_addons.layers import GroupNormalization
import os
from nn import get_timestep_embedding, downsample, upsample, resnet_block, attn_block

'''
h5 files for cifar and lsuns adapted from https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja. 
h5 files for celeba adapted from the provided pth file in https://github.com/ermongroup/ddim
The h5 files contain the exact same weights as the ones from the above links and contain only the exponentially moving averaged weights in the pth/ckpt files.
The h5 files are made to be compatible with tf 2.x / tfkeras 
h5 files were created by creating a tf.keras.Model object that defines the exact same layers as the original checkpoint in the exact same order, 
then calling model.set_weights on the properly ordered list of weights.
Parameter counts for each model:
cifar10: 35746307 params
lsun: 113673219 params
celeba: 78700803 params
'''
class DDPM_Checkpoint_Model(Model):
    def __init__(self, data_to_use, c=128):
        super().__init__() 
        self.data_to_use = data_to_use
        if data_to_use == 'cifar10':
            self.spatialres = 32
            chmul = [1, 2, 2, 2]
        elif 'lsun' in self.data_to_use:
            self.spatialres = 256
            chmul = [1, 1, 2, 2, 4, 4]
        else:
            raise NotImplementedError("dataset must be either cifar10, lsun_church. To load the Celeba model, instantiate a DDIM_Checkpoint_Model object.")

        self.c = c

        self.conv_in = Conv2D(c, 3, padding='same')
        self.conv_out = Conv2D(3, 3, padding='same')

        self.down_0 = [resnet_block(c*chmul[0],was_pytorch = False), resnet_block(c*chmul[0],was_pytorch = False), downsample(c*chmul[0], with_conv=True)]
        if self.spatialres == 32:
            self.attnsdown = [attn_block(c*chmul[1],was_pytorch = False), attn_block(c*chmul[1],was_pytorch = False)]
        self.down_1 = [resnet_block(c*chmul[1], use_nin_shortcut=(self.spatialres==32),was_pytorch = False), resnet_block(c*chmul[1],was_pytorch = False), downsample(c*chmul[1], with_conv=True)]
        self.down_2 = [resnet_block(c*chmul[2], use_nin_shortcut=(self.spatialres==256),was_pytorch = False), resnet_block(c*chmul[2],was_pytorch = False), downsample(c*chmul[2], with_conv=True)]
        self.down_3 = [resnet_block(c*chmul[3],was_pytorch = False), resnet_block(c*chmul[3],was_pytorch = False)]
        if self.spatialres == 256:
            self.down_3.append(downsample(c*chmul[3], with_conv=True))
            self.attnsdown = [attn_block(c*chmul[4],was_pytorch = False), attn_block(c*chmul[4],was_pytorch = False)]
            self.down_4 = [resnet_block(c*chmul[4], use_nin_shortcut=True,was_pytorch = False), resnet_block(c*chmul[4],was_pytorch = False), downsample(c*chmul[4], with_conv=True)]
            self.down_5 = [resnet_block(c*chmul[5],was_pytorch = False), resnet_block(c*chmul[5],was_pytorch = False)]
        
        self.mid1 = attn_block(c*chmul[-1],was_pytorch = False)
        self.mid0 = resnet_block(c*chmul[-1],was_pytorch = False)
        self.mid2 = resnet_block(c*chmul[-1],was_pytorch = False)

        self.norm_out = GroupNormalization(groups=32)
        self.temb = [Dense(c*4), Dense(c*4)]

        self.up_0 = [resnet_block(c*chmul[0], use_nin_shortcut=True,was_pytorch = False) for _ in range(3)]
        if self.spatialres == 32:
            self.attnsup = [attn_block(c*chmul[1],was_pytorch = False) for _ in range(3)]
        self.up_1 = [resnet_block(c*chmul[1], use_nin_shortcut=True,was_pytorch = False) for _ in range(3)] + [upsample(c*chmul[1], with_conv=True)]
        self.up_2 = [resnet_block(c*chmul[2], use_nin_shortcut=True,was_pytorch = False) for _ in range(3)] + [upsample(c*chmul[2], with_conv=True)]
        self.up_3 = [resnet_block(c*chmul[3], use_nin_shortcut=True,was_pytorch = False) for _ in range(3)] + [upsample(c*chmul[3], with_conv=True)]
        if self.spatialres == 256:
            self.attnsup = [attn_block(c*chmul[4], was_pytorch = False) for _ in range(3)]
            self.up_4 = [resnet_block(c*chmul[4], use_nin_shortcut=True,was_pytorch = False) for _ in range(3)] + [upsample(c*chmul[4], with_conv=True)]
            self.up_5 = [resnet_block(c*chmul[5], use_nin_shortcut=True,was_pytorch = False) for _ in range(3)] + [upsample(c*chmul[5], with_conv=True)]
    
    def get_pretrained_weights(self, weights_path):
        #note: model name must be specific. The model name must be model_tf2_{DATASET}.h5 e.g. model_tf2_lsun_church.h5
        self(tf.random.normal([1, self.spatialres, self.spatialres, 3]), tf.ones([1])) #builds model.
        if weights_path is not None:
            self.load_weights(weights_path) 
        

    def call(self, x, index):
        index = get_timestep_embedding(index, self.c)
        index = tf.nn.swish(self.temb[0](index))
        index = self.temb[1](index)

        x = self.conv_in(x)
        residuals = [tf.identity(x)]

        for block in self.down_0:
            x = block(x, index)
            residuals.append(x)
        
        if self.spatialres == 32:
            for i, block in enumerate(self.down_1):
                x = block(x, index)
                if i < 2: x = self.attnsdown[i](x, index)
                residuals.append(x)
        else:
            for block in self.down_1:
                x = block(x, index)
                residuals.append(x)

        for block in self.down_2:
            x = block(x, index)
            residuals.append(x)
        
        for block in self.down_3:
            x = block(x, index)
            residuals.append(x)

        if self.spatialres == 256:
            for i, block in enumerate(self.down_4):
                x = block(x, index)
                if i < 2: x = self.attnsdown[i](x, index)
                residuals.append(x)
            
            for block in self.down_5:
                x = block(x, index)
                residuals.append(x)
        x = self.mid0(x, index)
        x = self.mid1(x, index)
        x = self.mid2(x, index)
        
        if self.spatialres == 256:
            for i, block in enumerate(self.up_5):
                if i<3:
                    x = tf.concat([x, residuals.pop()], axis=-1)
                x = block(x, index)

            for i, block in enumerate(self.up_4):
                if i<3:
                    x = tf.concat([x, residuals.pop()], axis=-1)
                x = block(x, index)
                if i<3:
                    x = self.attnsup[i](x, index)

        for i, block in enumerate(self.up_3):
            if i<3:
                x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)

        for i, block in enumerate(self.up_2):
            if i<3:
                x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)
            
        if self.spatialres == 32:
            for i, block in enumerate(self.up_1):
                if i<3:
                    x = tf.concat([x, residuals.pop()], axis=-1)
                x = block(x, index)
                if i<3:
                    x = self.attnsup[i](x, index)
        else:
            for i, block in enumerate(self.up_1):
                if i<3:
                    x = tf.concat([x, residuals.pop()], axis=-1)
                x = block(x, index)

        for i, block in enumerate(self.up_0):
            x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)

        x = tf.nn.swish(self.norm_out(x))
        x = self.conv_out(x)
        
        return x

class DDIM_Checkpoint_Model(Model):
    def __init__(self, c=128):
        super().__init__() 
        chmul = [1, 2, 2, 2, 4]
        self.c = c

        self.temb = [Dense(c*4), Dense(c*4)]
        self.conv_in = Conv2D(c, 3, padding='same')

        self.down_0 = [resnet_block(c*chmul[0],was_pytorch = True), resnet_block(c*chmul[0],was_pytorch = True), downsample(c*chmul[0], with_conv=True)]
        self.down_1 = [resnet_block(c*chmul[1], use_nin_shortcut=True,was_pytorch = True), resnet_block(c*chmul[1],was_pytorch = True), downsample(c*chmul[1], with_conv=True)]
        self.down_2 = [resnet_block(c*chmul[2],was_pytorch = True), resnet_block(c*chmul[2],was_pytorch = True)]
        self.attnsdown = [attn_block(c*chmul[1],was_pytorch = True), attn_block(c*chmul[1],was_pytorch = True)]
        self.downsample2 = downsample(c*chmul[2], with_conv=True)
        self.down_3 = [resnet_block(c*chmul[3],was_pytorch = True), resnet_block(c*chmul[3],was_pytorch = True), downsample(c*chmul[3], with_conv=True)]
        self.down_4 = [resnet_block(c*chmul[4], use_nin_shortcut=True,was_pytorch = True), resnet_block(c*chmul[4],was_pytorch = True)]
        
        self.mids = [resnet_block(c*chmul[-1],was_pytorch = True), attn_block(c*chmul[-1],was_pytorch = True),resnet_block(c*chmul[-1],was_pytorch = True)]

        self.up_0 = [resnet_block(c*chmul[0], use_nin_shortcut=True,was_pytorch = True) for _ in range(3)]
        self.up_1 = [resnet_block(c*chmul[1], use_nin_shortcut=True,was_pytorch = True) for _ in range(3)] + [upsample(c*chmul[1], with_conv=True)]
        self.up_2 = [resnet_block(c*chmul[2], use_nin_shortcut=True,was_pytorch = True) for _ in range(3)]
        self.attnsup = [attn_block(c*chmul[2],was_pytorch = True) for _ in range(3)]
        self.upsample2 = upsample(c*chmul[2], with_conv=True)
        self.up_3 = [resnet_block(c*chmul[3], use_nin_shortcut=True,was_pytorch = True) for _ in range(3)] + [upsample(c*chmul[3], with_conv=True)]
        self.up_4 = [resnet_block(c*chmul[4], use_nin_shortcut=True,was_pytorch = True) for _ in range(3)] + [upsample(c*chmul[4], with_conv=True)]
        

        self.norm_out = GroupNormalization(groups=32)
        self.conv_out = Conv2D(3, 3, padding='same')

    def get_pretrained_weights(self, weights_path):
        #note: model name must be specific. The model name must be model_tf2_{DATASET}.h5 e.g. model_tf2_lsun_church.h5
        self(tf.random.normal([1, 64, 64, 3]), tf.ones([1])) #builds model.
        if weights_path is not None:
            self.load_weights(weights_path) 
        

    def call(self, x, index):
        index = get_timestep_embedding(index, self.c)
        index = tf.nn.swish(self.temb[0](index))
        index = self.temb[1](index)

        x = self.conv_in(x)
        residuals = [tf.identity(x)]

        for block in self.down_0:
            x = block(x, index)
            residuals.append(x)

        for block in self.down_1:
            x = block(x, index)
            residuals.append(x)

        for i, block in enumerate(self.down_2):
            x = block(x, index)
            if i < 2: x = self.attnsdown[i](x, index)
            residuals.append(x)
        
        x = self.downsample2(x, index)
        residuals.append(x)
        
        for block in self.down_3:
            x = block(x, index)
            residuals.append(x)

        for block in self.down_4:
            x = block(x, index)
            residuals.append(x)

        for block in self.mids:
            x = block(x, index)
        
        for i, block in enumerate(self.up_4):
            if i<3:
                x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)

        for i, block in enumerate(self.up_3):
            if i<3:
                x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)

        for i, block in enumerate(self.up_2):
            x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)
            x = self.attnsup[i](x, index)
        
        x = self.upsample2(x, index)
        
        for i, block in enumerate(self.up_1):
            if i<3:
                x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)

        for i, block in enumerate(self.up_0):
            x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)

        x = tf.nn.swish(self.norm_out(x))
        x = self.conv_out(x)
        
        return x

'''
The generative model that we use in our experiments. It takes a single input, X_T ~ N(0, 1) and returns the predicted image  X_0 in a single step.
It loads the converted weights of the original DDPM/DDIM model (from the h5 files)
The model takes two arguments: 
data_to_use - the dataset that it is trained on
model_path - the location of the pretrained weights in h5 format. can be None.
run_ddim_step method executes 1 step of the DDIM process.
'''

class Onestep_Model(Model):
    def __init__(self, data_to_use, model_path):
        super().__init__()
        self.data_to_use = data_to_use
        if data_to_use == 'cifar10':
            self.pretrained_model = DDPM_Checkpoint_Model(data_to_use)
            self.spatialres = 32
        elif 'lsun' in data_to_use:
            self.pretrained_model = DDPM_Checkpoint_Model(data_to_use)
            self.spatialres = 256
        elif data_to_use == 'celeba':
            self.pretrained_model = DDIM_Checkpoint_Model()
            self.spatialres = 64
        else:
            raise NotImplementedError

        self.pretrained_model.get_pretrained_weights(model_path)

    def call(self, z):
        #uses the highest index seen by the pretrained model.
        inp = tf.identity(z)
        index = tf.ones_like(z[:, 0, 0, 0]) * 999.
        x = self.pretrained_model(z, index) 
        #this layers output can be thought of as returning a prediction for epsilon.
        pred_y = inp - x
        return pred_y
    
    def run_ddim_step(self, xt, index, alpha, alpha_next):
        eps = self.pretrained_model(xt, index)
        x_t_minus1 = tf.sqrt(alpha_next) * (xt - tf.sqrt(1-alpha)*eps) / tf.sqrt(alpha)
        x_t_minus1 += tf.sqrt(1-alpha_next) * eps 
        return x_t_minus1