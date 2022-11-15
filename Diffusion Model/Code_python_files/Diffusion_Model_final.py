#import relevant libraries
import tensorflow as tf
import math
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  AveragePooling2D, UpSampling2D, BatchNormalization, Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Reshape, Conv2DTranspose, Embedding, Concatenate, Multiply, Add, Lambda
from tensorflow.keras.layers.experimental.preprocessing import Normalization 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
import os
from U_net_library import *

# optimization
class UNET(object):
    
    def __init__(self, n_layers, n_filters, inp_shape):
        #the paper suggests f_max to be log(T) 
        
        self.arch_hyperparameters = {'l2_conv': 0.0001
                                }
        self.inp_shape = inp_shape
        #using https://arxiv.org/abs/2010.02502
        #exact embeddings used
        
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.unet_model = None
    
    #before you begin idea is to add an embedding channel to make the data (None, m, n, 4), to propagate this information,
    #a linear 1x1 convolution is done in the image, followed by a (None, m, n, 3)
    def get_sinusoidal_embedding_channel(self, t_var):
        #fix the omegas and the number of channels to add
        omegas = 2.0*math.pi*tf.exp(tf.linspace(0.0, math.log(1000), 8))
        
        sinusoid_emb = Lambda(lambda x: tf.concat([tf.sin(x[0] * x[1]), tf.cos(x[0] * x[1])], axis=3), name='Embedding_layer')([omegas, t_var])
        return sinusoid_emb

   
    def conv_block_ds(self, inp, layer, n_filters):
        #using standard u-net architecture
        conv_op = Conv2D(n_filters, (3,3), strides=(1,1), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'ds_conv_'+str(layer)+'_a')(inp)
        conv_op = BatchNormalization(name='ds_batch_norm_l'+str(layer)+"_a")(conv_op)
        conv_op = Conv2D(n_filters, (3,3), strides=(1,1), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'ds_conv_'+str(layer)+'_b')(conv_op)
        conv_op = BatchNormalization(name='ds_batch_norm_l'+str(layer)+"_b")(conv_op)
        #maxpool:
        mpool = MaxPool2D((2,2), strides=(2,2), padding='same', name='ds_maxpool_'+str(layer))(conv_op)
        
        return conv_op, mpool

    def conv_block_us(self, inp, inp_skip, layer, n_filter1, n_filter2):
        #upsample:
        inp = Conv2DTranspose(inp.shape[-1], (3,3), strides=(2,2), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'convTrans_'+str(layer))(inp)
        inp = BatchNormalization(name='us_batch_norm_l'+str(layer)+'_a')(inp)
        #inp resize
        inp = Lambda(lambda x: tf.image.resize_with_crop_or_pad(x[0], x[1].shape[1], x[1].shape[2]), name='us_resize_'+str(layer))([inp, inp_skip])
        u_conv_out = Concatenate(name='concat_'+str(layer))([inp, inp_skip])
        u_conv_op = Conv2D(n_filter1, (3,3), strides=(1,1), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'us_conv_'+str(layer)+'_a')(u_conv_out)
        u_conv_op = BatchNormalization(name='us_batch_norm_l'+str(layer)+'_b')(u_conv_op)
        u_conv_op = Conv2D(n_filter2, (3,3), strides=(1,1), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'us_conv_'+str(layer)+'_b')(u_conv_op)
        u_conv_op = BatchNormalization(name='us_batch_norm_l'+str(layer)+'_c')(u_conv_op)
        return u_conv_op

    def make_UNET(self):
        inp = Input(shape=self.inp_shape, name='unet_input')
        t = Input(shape=(1,1,1), name='t_vars_proxy')
        #get time embedding
        t_emb = self.get_sinusoidal_embedding_channel(t)
        #make another channel of it
        t_emb = UpSampling2D(size=self.inp_shape[0], interpolation="nearest")(t_emb)
        for_dat = Concatenate(name="input_time_added")([inp, t_emb])

        for_dat = Conv2D(self.n_filters[0], (1,1), strides=(1,1), activation=None, padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'inp_mix_time_info')(for_dat)
        
        conv_ds = []
        #starting the forward pass

        for i in range(self.n_layers-1):
            conv_op, for_dat = self.conv_block_ds(for_dat, i, self.n_filters[i+1])
            conv_ds.append(conv_op)

        for_dat = Conv2D(self.n_filters[self.n_layers-1], (3,3), strides=(1,1), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'conv_layer_withoutskip')(for_dat)
        for_dat = BatchNormalization(name='bn_withoutskip')(for_dat)
        #this is the upsampling part
        for i in range(self.n_layers-1):
            for_dat = self.conv_block_us(for_dat, conv_ds[self.n_layers - i - 2], i, self.n_filters[self.n_layers - i - 1], self.n_filters[self.n_layers - i - 2])

        #for_dat = Concatenate(name="output_time_added")([for_dat, t_emb])
        for_dat = Conv2D(self.n_filters[0], (1,1), strides=(1,1), padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'scale_output_final')(for_dat)
        self.unet_model = Model(inputs=[inp, t], outputs=for_dat, name='Unet_model')
        
        self.unet_model.compile()
        #self.unet_model.summary()
        
    def make_UNET_yappended(self, n_attr):
        inp = Input(shape=self.inp_shape, name='unet_input')
        t = Input(shape=(1,1,1), name='t_vars_proxy')
        y = Input(shape=(n_attr, 1, 1), name='labels')
        y_upsamp = UpSampling2D(size=(int(self.inp_shape[0]/n_attr), self.inp_shape[1]), interpolation="nearest")(y)
        y_upsamp = Lambda(lambda x: tf.image.resize_with_crop_or_pad(x[0], x[1].shape[1], x[1].shape[2]), name='y_labels_resize')([y_upsamp, inp])
        #get time embedding
        t_emb = self.get_sinusoidal_embedding_channel(t)
        #make another channel of it
        t_emb = UpSampling2D(size=(self.inp_shape[0], self.inp_shape[1]), interpolation="nearest")(t_emb)
        for_dat = Concatenate(name="input_time_y_added")([inp, t_emb, y_upsamp])

        for_dat = Conv2D(self.n_filters[0], (1,1), strides=(1,1), activation=None, padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'inp_mix_time_info')(for_dat)
        
        conv_ds = []
        #starting the forward pass

        for i in range(self.n_layers-1):
            conv_op, for_dat = self.conv_block_ds(for_dat, i, self.n_filters[i+1])
            conv_ds.append(conv_op)

        for_dat = Conv2D(self.n_filters[self.n_layers-1], (3,3), strides=(1,1), activation='elu', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'conv_layer_withoutskip')(for_dat)
        for_dat = BatchNormalization(name='bn_withoutskip')(for_dat)
        #this is the upsampling part
        for i in range(self.n_layers-1):
            for_dat = self.conv_block_us(for_dat, conv_ds[self.n_layers - i - 2], i, self.n_filters[self.n_layers - i - 1], self.n_filters[self.n_layers - i - 2])

        #for_dat = Concatenate(name="output_time_added")([for_dat, t_emb])
        for_dat = Conv2D(self.n_filters[0], (1,1), strides=(1,1), padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'scale_output_final')(for_dat)
        self.unet_model = Model(inputs=[inp, t, y], outputs=for_dat, name='Unet_model')
        
        self.unet_model.compile()
        #self.unet_model.summary()
        

    
class Diffusion_Model_modified_schedule(object):
    def __init__(self, input_shape, T, inbuilt_model = False):
        #T is the mixing time
        self.input_shape = input_shape
        self.inbuilt_model = inbuilt_model
        self.arch_hyperparameters = {
                                     'batch_size': 24,
                                     'lr': 0.001,
                                     'T': T,
                                     'angle_ends': np.arccos(np.array([0.98, 0.02])),
                                     'beta_t': None,
                                     'alpha_t': None
                                    }
       
        self.calculate_alphas_sinusoidal_schedule()
        #trying to learn the sigma instead of using a variance
        self.sigma_q_t_2 = Normalization()
        
        self.mu = None
        self.var = None
        self.u_net_diff = None
        self.optimizer = None
        
    
    def make_model(self):
        if self.inbuilt_model == False:
            u_net_diff = UNET(5, [3, 32, 64, 96, 128], self.input_shape)
            u_net_diff.make_UNET()
            #assign the models
            self.u_net_diff = u_net_diff.unet_model
        else:
            # the same structure as my u_net: using 3 conv layers in up and downsample
            self.u_net_diff = get_network(self.input_shape[0], [32, 64, 96, 128], 2)
        #self.u_net_diff.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])
        
        self.u_net_diff.compile()
        self.u_net_diff.summary()
        
        
    def save_model(self, path):
        self.u_net_diff.save(path+"unet_diff")
        
        z = np.array([self.sigma_q_t_2.mean, self.sigma_q_t_2.variance])
        with open(path+"norm.npy", 'wb') as fptr:
           np.save(fptr, z)
        
    def load_mymodel(self, path):
        print("Loading Model")
        self.u_net_diff = load_model(path+"unet_diff")
        self.u_net_diff.compile()
        self.u_net_diff.summary()
    
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])
        #load the embeddings
        
        z = np.load(path+"norm.npy")
        self.mu = z[0]
        self.var = z[1]
        self.sigma_q_t_2.mean = z[0]
        self.sigma_q_t_2.variance = z[1]
    
    def calculate_alphas_sinusoidal_schedule(self):
        a_0 = self.arch_hyperparameters['angle_ends'][0]
        grad = (self.arch_hyperparameters['angle_ends'][1] - self.arch_hyperparameters['angle_ends'][0])/self.arch_hyperparameters['T']
        
        #root(a_t_bar) = cos x  (remember these are bars)
        #root(b_t_bar) = root(1-a_t)
        t = np.arange(self.arch_hyperparameters['T']+1).astype('float32')
        self.arch_hyperparameters['alpha_t'] = np.cos(a_0 + grad*t)
        self.arch_hyperparameters['beta_t'] = np.sin(a_0 + grad*t)
        
        
    def X_t(self, data_X_0, t_vector):
        #returns the noise added images
        
        u_xt_giv_x0 = np.expand_dims(self.arch_hyperparameters['alpha_t'][t_vector], axis=[1,2,3])*data_X_0
        sigma_xt_giv_x0 = np.expand_dims(self.arch_hyperparameters['beta_t'][t_vector], axis=[1,2,3]) 
        
        eps = np.random.normal(size=data_X_0.shape) 
        
        X_t = u_xt_giv_x0 + sigma_xt_giv_x0*eps
        return X_t, eps
    
    def loss(self, batch):
        
        #get all X_ts random selected for each image
        #generate a list of random numbers [2, T]*num_image*number_estimat3 (=1 here) for each, 1 is the hack to learn the last step too
        ts = np.random.randint(1, self.arch_hyperparameters['T']+1, size=batch.shape[0])
        
        batch = self.sigma_q_t_2(batch, training=False)
        batch_appended, noise = self.X_t(batch, ts)
        
        ep_theta_ts = self.u_net_diff([batch_appended, np.expand_dims(self.arch_hyperparameters['beta_t'][ts]**2, axis=[1, 2, 3])])
        #MSE loss
        #mse_diff = tf.reduce_sum(((ep_theta_ts - noise)**2)/2, axis=[1, 2, 3])
        #loss_diff = tf.reduce_mean(mse_diff)
        #MAE loss, it is seen to work better
        mae_diff = tf.reduce_sum(tf.math.abs(ep_theta_ts - noise), axis=[1, 2, 3])
        loss_diff = tf.reduce_mean(mae_diff)
        del(batch_appended)
        
        #return loss_recon, loss_diff
        return loss_diff
        
    def back_prop_noise_formualation(self, batch):
        #get all X_ts random selected for each image
        #generate a list of random numbers [1, T]*num_image*number_estimate for each [all diffusion model algorithms use this trick]
        
        
        #I am minimizing the negative of the ELBO
        #train the model
        with tf.GradientTape(persistent=True) as tape:
            batch = self.sigma_q_t_2(batch, training=True)
            ts = np.random.randint(1, self.arch_hyperparameters['T']+1, size=batch.shape[0])
            #repeat the first dimension t times (to estimate the sum 2 to T)
            batch, noise = self.X_t(batch, ts)
            
            ep_theta_ts = self.u_net_diff([batch, np.expand_dims(self.arch_hyperparameters['beta_t'][ts]**2, axis=[1, 2, 3])])
            #MSE loss
            #mse_diff = tf.reduce_sum(((ep_theta_ts - noise)**2)/2, axis=[1, 2, 3])
            #loss_diff = tf.reduce_mean(mse_diff)
            #MAE loss
            mae_diff = tf.reduce_sum(tf.math.abs(ep_theta_ts - noise), axis=[1, 2, 3])
            loss_diff = tf.reduce_mean(mae_diff)
        
        #calculate gradient
        grad_diff = tape.gradient(loss_diff, self.u_net_diff.trainable_variables)
        
        #gradient descent
        self.optimizer.apply_gradients(zip(grad_diff, self.u_net_diff.trainable_variables))
        #this is done for numerical stability from sudden jumps: exponential moving average mentioned in the paper
        
        
    
    
    def sample_images(self, num_img, inp_imgs=None, from_img=False):
        sampled_langevian_steps = []
        sample_every = int(0.1*self.arch_hyperparameters['T'])
        if from_img == True:
          X_t, _ = self.X_t(sel.prep(inp_images), (np.ones(inp_imgs.shape[0])*self.arch_hyperparameters['T']).astype('int'))
        else:
          #vectorized implementation for the entire batch
          X_t = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2])).astype('float32')
        
        for t in range(self.arch_hyperparameters['T'], 0, -1):
            if t%sample_every == 0:
                #use the normalizaton layer
                app = self.sigma_q_t_2.mean + X_t*tf.sqrt(self.sigma_q_t_2.variance) 
                app = tf.clip_by_value(app, 0, 1)
                app = (app.numpy()*255).astype('uint8')
                sampled_langevian_steps.append(app)
            
            z = 0
            if t>1:
                z = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            
            
            
            ts = t*np.ones((num_img, 1, 1, 1), dtype='int32')
            
            eps_theta_X_t = self.u_net_diff([X_t, self.arch_hyperparameters['beta_t'][ts]**2])
            
            w1 = self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            
            w2 = self.arch_hyperparameters['beta_t'][ts]*self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            w3 = self.arch_hyperparameters['beta_t'][ts-1]
            #next step is the noise removed from the current step and the noise added to it
            #X_t = w1*X_t - w2*eps_theta_X_t + w3*z
            X_t = w1*X_t - w2*eps_theta_X_t + w3*eps_theta_X_t      
        #final image
        app = self.sigma_q_t_2.mean + X_t*tf.sqrt(self.sigma_q_t_2.variance) 
        app = tf.clip_by_value(app, 0, 1)
        app = (app.numpy()*255).astype('uint8')
        sampled_langevian_steps.append(app)
        
        return sampled_langevian_steps
    
    def sample_images2(self, num_img):
        
        
        X_t = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2])).astype('float32')
        
        for t in range(self.arch_hyperparameters['T'], 0, -1):
            z = 0
            if t>1:
                z = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            
            ts = t*np.ones((num_img, 1, 1, 1), dtype='int32')
            
            eps_theta_X_t = self.u_net_diff([X_t, self.arch_hyperparameters['beta_t'][ts]**2])
            
            w1 = self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            
            w2 = self.arch_hyperparameters['beta_t'][ts]*self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            w3 = self.arch_hyperparameters['beta_t'][ts-1]
            #next step is the noise removed from the current step and the noise added to it
            #X_t = w1*X_t - w2*eps_theta_X_t + w3*z
            X_t = w1*X_t - w2*eps_theta_X_t + w3*eps_theta_X_t      
        
        #final image
        app = self.sigma_q_t_2.mean + X_t*tf.sqrt(self.sigma_q_t_2.variance) 
        app = tf.clip_by_value(app, 0, 1)
        app = (app.numpy()*255).astype('uint8')
        return app
        
    def train_diff_model_noise_formulation(self, data, n_epochs):
        
        ep_length = int(data.shape[0]/self.arch_hyperparameters['batch_size'])
        #loss_history = {
        #                'recons': [],
        #                'diff': []
        #                }
        
        loss = []
        for i in range(n_epochs):
            samples = data[np.random.choice(data.shape[0], self.arch_hyperparameters['batch_size'], replace=False)]
            #r, d = self.loss(samples)
            d = self.loss(samples)
            
            #loss_history['recons'].append(r)
            #loss_history['diff'].append(d)
            loss.append(d)
            if i%2 == 0:
                #print("Losses at epoch {}, reconstruction: {} Denoising: {} ".format(i, r, d))
                print("Losses at epoch {}, Denoising: {} ".format(i, d))
                            
            np.random.shuffle(data)
            for j in range(ep_length):
                if j == ep_length - 1:
                    batch_real = data[j*self.arch_hyperparameters['batch_size']:data.shape[0]]
                else:
                    batch_real = data[j*(self.arch_hyperparameters['batch_size']):(j+1)*(self.arch_hyperparameters['batch_size'])]
                
                self.back_prop_noise_formualation(batch_real)
        
        return loss


class Diffusion_Model_classifier_guidance(object):
    def __init__(self, input_shape, T, n_attr, inbuilt_model = False):
        #T is the mixing time
        self.input_shape = input_shape
        self.inbuilt_model = inbuilt_model
        self.num_attrs = n_attr
        
        self.arch_hyperparameters = {
                                     'l2_dense': 0.001,
                                     'dropout': 0.2,
                                     'l2_conv': 0.0001,
                                     'batch_size': 24,
                                     'lr_class': 0.001,
                                     'lr_unet': 0.0001,
                                     'T': T,
                                     'angle_ends': np.arccos(np.array([0.98, 0.02])),
                                     'beta_t': None,
                                     'alpha_t': None,
                                     'weight_nu': 10
                                    }
       
        self.calculate_alphas_sinusoidal_schedule()
        #trying to learn the sigma instead of using a variance
        self.sigma_q_t_2 = Normalization()
        
        self.mu = None
        self.var = None
        self.u_net_diff = None
        self.classifier = None
        self.optimizer_u = None
        self.optimizer_c = None
    
    def make_classifier(self):
        #For a strong classifier to help, a Resnet 50 is used
        from tensorflow.keras.applications.resnet50 import ResNet50
        
        #resnet block
        inp = Input(shape=self.input_shape, name='classifier_input')
        rnet50_model = ResNet50(include_top=False, weights=None, input_tensor=inp, input_shape=self.input_shape, pooling='avg')
        res_out = rnet50_model(inp)
        
        #my fully connected blocks
        x = Dropout(self.arch_hyperparameters['dropout'], name = 'Dropout_final')(res_out)
        #form 2048-128
        x = Dense(128, activation ='elu', kernel_regularizer=l2(self.arch_hyperparameters['l2_dense']), name = 'Dense')(x)
        
        class_out = Dense(self.num_attrs, activation ='sigmoid', kernel_regularizer=l2(self.arch_hyperparameters['l2_dense']), name = 'Dense_out')(x)
        class_model = Model(inputs=inp, outputs=class_out, name='classifier_guide')
        return class_model
    
    def make_model(self):
        if self.inbuilt_model == False:
            u_net_diff = UNET(5, [3, 32, 64, 96, 128], self.input_shape)
            u_net_diff.make_UNET_yappended(self.num_attrs)
            #assign the models
            self.u_net_diff = u_net_diff.unet_model
        else:
            # the same structure as my u_net: using 3 conv layers in up and downsample
            self.u_net_diff = get_network_classifier_guidance(self.input_shape[0], [32, 64, 96, 128], 3, self.num_attrs)
        #self.u_net_diff.summary()
        
        #get the classifier
        self.classifier = self.make_classifier()
        self.classifier.compile()
        self.classifier.summary()
        
        self.optimizer_u = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr_unet'])
        self.optimizer_c = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr_class'])
        
        self.u_net_diff.compile()
        self.u_net_diff.summary()
        
        
    def save_model(self, path):
        self.u_net_diff.save(path+"unet_diff")
        self.classifier.save(path+"classifier")
        z = np.array([self.sigma_q_t_2.mean, self.sigma_q_t_2.variance])
        with open(path+"norm.npy", 'wb') as fptr:
           np.save(fptr, z)
        
    def load_mymodel(self, path):
        print("Loading Model")
        #load the u-net
        self.u_net_diff = load_model(path+"unet_diff")
        self.u_net_diff.compile()
        self.u_net_diff.summary()
        #load the classifier
        self.classifier = load_model(path+"classifier")
        self.classifier.compile()
        self.classifier.summary()
        
        self.optimizer_u = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr_unet'])
        self.optimizer_c = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr_class'])
        #load the embeddings
        
        z = np.load(path+"norm.npy")
        self.mu = z[0]
        self.var = z[1]
        self.sigma_q_t_2.mean = z[0]
        self.sigma_q_t_2.variance = z[1]
    
    def calculate_alphas_sinusoidal_schedule(self):
        a_0 = self.arch_hyperparameters['angle_ends'][0]
        grad = (self.arch_hyperparameters['angle_ends'][1] - self.arch_hyperparameters['angle_ends'][0])/self.arch_hyperparameters['T']
        
        #root(a_t_bar) = cos x  (remember these are bars)
        #root(b_t_bar) = root(1-a_t)
        t = np.arange(self.arch_hyperparameters['T']+1).astype('float32')
        self.arch_hyperparameters['alpha_t'] = np.cos(a_0 + grad*t)
        self.arch_hyperparameters['beta_t'] = np.sin(a_0 + grad*t)
        
        
    def X_t(self, data_X_0, t_vector):
        #returns the noise added images
        
        u_xt_giv_x0 = np.expand_dims(self.arch_hyperparameters['alpha_t'][t_vector], axis=[1,2,3])*data_X_0
        sigma_xt_giv_x0 = np.expand_dims(self.arch_hyperparameters['beta_t'][t_vector], axis=[1,2,3]) 
        
        eps = np.random.normal(size=data_X_0.shape) 
        
        X_t = u_xt_giv_x0 + sigma_xt_giv_x0*eps
        return X_t, eps
    
    def loss(self, batch, y):
        
        #get all X_ts random selected for each image
        #generate a list of random numbers [2, T]*num_image*number_estimat3 (=1 here) for each, 1 is the hack to learn the last step too
        ts = np.random.randint(1, self.arch_hyperparameters['T']+1, size=batch.shape[0])
        
        batch = self.sigma_q_t_2(batch, training=False)
        batch_appended, noise = self.X_t(batch, ts)
        
        
        ep_theta_ts = self.u_net_diff([batch_appended, np.expand_dims(self.arch_hyperparameters['beta_t'][ts]**2, axis=[1, 2, 3]), np.expand_dims(y, axis=[2, 3])])
        #MSE loss
        #mse_diff = tf.reduce_sum(((ep_theta_ts - noise)**2)/2, axis=[1, 2, 3])
        #loss_diff = tf.reduce_mean(mse_diff)
        #MAE loss, it is seen to work better
        mae_diff = tf.reduce_sum(tf.math.abs(ep_theta_ts - noise), axis=[1, 2, 3])
        loss_diff = tf.reduce_mean(mae_diff)
        del batch_appended
        
        #predict the labels
        pred_labels = self.classifier(batch)
        #this is a binary cross entropy loss for each of the output label as it is a multiclass classification problem
        loss_class = tf.keras.losses.BinaryCrossentropy()(y, pred_labels)
        #per_example loss
        loss_class = self.arch_hyperparameters['weight_nu']*loss_class*y.shape[0]
        #this is the classifier accuracy: overall (collapsing all the attributes)
        y_pred_n = (pred_labels>0.5).numpy()
        y_n = y.astype('uint8')
        acc_class = np.sum(y_pred_n==y_n)/(y.shape[0]*y.shape[1])
        
        return loss_diff, loss_class, acc_class
        
    def back_prop_noise_formualation(self, batch, y):
        #get all X_ts random selected for each image
        #generate a list of random numbers [1, T]*num_image*number_estimate for each [all diffusion model algorithms use this trick]
        
        
        #I am minimizing the negative of the ELBO
        #train the model
        with tf.GradientTape(persistent=True) as tape:
            batch = self.sigma_q_t_2(batch, training=True)
            ts = np.random.randint(1, self.arch_hyperparameters['T']+1, size=batch.shape[0])
            #repeat the first dimension t times (to estimate the sum 2 to T)
            batch, noise = self.X_t(batch, ts)
            
            ep_theta_ts = self.u_net_diff([batch, np.expand_dims(self.arch_hyperparameters['beta_t'][ts]**2, axis=[1, 2, 3]), np.expand_dims(y, axis=[2, 3])])
            #MSE loss
            #mse_diff = tf.reduce_sum(((ep_theta_ts - noise)**2)/2, axis=[1, 2, 3])
            #loss_diff = tf.reduce_mean(mse_diff)
            #MAE loss
            mae_diff = tf.reduce_sum(tf.math.abs(ep_theta_ts - noise), axis=[1, 2, 3])
            loss_diff = tf.reduce_mean(mae_diff)
            
        #calculate gradient
        grad_diff = tape.gradient(loss_diff, self.u_net_diff.trainable_variables)
        #gradient descent
        self.optimizer_u.apply_gradients(zip(grad_diff, self.u_net_diff.trainable_variables))
        #this is done for numerical stability from sudden jumps: exponential moving average mentioned in the paper
        del grad_diff
        del ep_theta_ts
        del tape
        
        with tf.GradientTape(persistent=True) as tape:
            pred_labels = self.classifier(batch)
            #this is a binary cross entropy loss for each of the output label as it is a multiclass classification problem
            loss_class = tf.keras.losses.BinaryCrossentropy()(y, pred_labels)
            #per_example loss
            loss_class = self.arch_hyperparameters['weight_nu']*loss_class*y.shape[0]
        
        grad_class = tape.gradient(loss_class, self.classifier.trainable_variables)
        #gradient descent
        self.optimizer_c.apply_gradients(zip(grad_class, self.classifier.trainable_variables))
        #this is done for numerical stability from sudden jumps: exponential moving average mentioned in the paper
        del grad_class
        del tape
    
    def get_score_classifier(self, X_t, y):
        inp = tf.constant(X_t)    
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inp)      
            pred_labels = self.classifier(inp)
            
            #this is a binary cross entropy loss for each of the output label as it is a multiclass classification problem
            loss_class = tf.keras.losses.BinaryCrossentropy()(y, pred_labels)
            #per_example loss
            loss_class = self.arch_hyperparameters['weight_nu']*loss_class*y.shape[0]
        
        score = tape.gradient(loss_class, inp)
        del tape
        return score
    
    def sample_images(self, num_img, labels, inp_imgs=None, from_img=False):
        sampled_langevian_steps = []
        sample_every = int(0.1*self.arch_hyperparameters['T'])
        if from_img == True:
          X_t, _ = self.X_t(sel.prep(inp_images), (np.ones(inp_imgs.shape[0])*self.arch_hyperparameters['T']).astype('int'))
        else:
          #vectorized implementation for the entire batch
          X_t = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2])).astype('float32')
        
        for t in range(self.arch_hyperparameters['T'], 0, -1):
            if t%sample_every == 0:
                #use the normalizaton layer
                app = self.sigma_q_t_2.mean + X_t*tf.sqrt(self.sigma_q_t_2.variance) 
                app = tf.clip_by_value(app, 0, 1)
                app = (app.numpy()*255).astype('uint8')
                sampled_langevian_steps.append(app)
            
            z = 0
            if t>1:
                z = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
                
            
            
            ts = t*np.ones((num_img, 1, 1, 1), dtype='int32')
            
            eps_theta_X_t = self.u_net_diff([X_t, self.arch_hyperparameters['beta_t'][ts]**2, np.expand_dims(labels, axis=[2, 3])])
            class_score = self.get_score_classifier(X_t, labels)
            
            w1 = self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            
            w2 = self.arch_hyperparameters['beta_t'][ts]*self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            w3 = self.arch_hyperparameters['beta_t'][ts-1]
            #next step is the noise removed from the current step and the noise added to it
            #X_t = w1*X_t - w2*eps_theta_X_t + w3*z
            X_t = w1*X_t - w2*(eps_theta_X_t + class_score) + w3*eps_theta_X_t      
        #final image
        app = self.sigma_q_t_2.mean + X_t*tf.sqrt(self.sigma_q_t_2.variance) 
        app = tf.clip_by_value(app, 0, 1)
        app = (app.numpy()*255).astype('uint8')
        sampled_langevian_steps.append(app)
        
        return sampled_langevian_steps
    
    def sample_images2(self, num_img, labels):
        
        
        X_t = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2])).astype('float32')
        
        for t in range(self.arch_hyperparameters['T'], 0, -1):
            z = 0
            if t>1:
                z = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            
            ts = t*np.ones((num_img, 1, 1, 1), dtype='int32')
            
            eps_theta_X_t = self.u_net_diff([X_t, self.arch_hyperparameters['beta_t'][ts]**2, np.expand_dims(labels, axis=[2, 3])])
            class_score = self.get_score_classifier(X_t, labels)
            
            w1 = self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            
            w2 = self.arch_hyperparameters['beta_t'][ts]*self.arch_hyperparameters['alpha_t'][ts-1]/self.arch_hyperparameters['alpha_t'][ts]
            w3 = self.arch_hyperparameters['beta_t'][ts-1]
            #next step is the noise removed from the current step and the noise added to it
            #X_t = w1*X_t - w2*eps_theta_X_t + w3*z
            X_t = w1*X_t - w2*(eps_theta_X_t + class_score) + w3*eps_theta_X_t      
        
        #final image
        app = self.sigma_q_t_2.mean + X_t*tf.sqrt(self.sigma_q_t_2.variance) 
        app = tf.clip_by_value(app, 0, 1)
        app = (app.numpy()*255).astype('uint8')
        return app
        
        
    def train_diff_model_noise_formulation(self, data, labels, n_epochs):
        
        ep_length = int(data.shape[0]/self.arch_hyperparameters['batch_size'])
        loss_history = {
                        'classifier': [],
                        'diff': [],
                        }
        acc_history = []
        for i in range(n_epochs):
            ch = np.random.choice(data.shape[0], self.arch_hyperparameters['batch_size'], replace=False)
            samples = data[ch]
            y_samples = labels[ch]
            #r, d = self.loss(samples)
            d, c, ac = self.loss(samples, y_samples)
            
            acc_history.append(ac)
            
            loss_history['classifier'].append(c)
            loss_history['diff'].append(d)
            
            if i%2 == 0:
                print("Losses at epoch {}, Denoising: {}, Classifier, loss: {}, Accuracy= {}".format(i, d, c, ac))
            
                            
            shuff = np.arange(data.shape[0])
            np.random.shuffle(shuff)
            
            data = data[shuff]
            labels = labels[shuff]
            
            for j in range(ep_length):
                if j == ep_length - 1:
                    batch_real = data[j*self.arch_hyperparameters['batch_size']:data.shape[0]]
                    labels_real = labels[j*self.arch_hyperparameters['batch_size']:data.shape[0]]
                else:
                    batch_real = data[j*(self.arch_hyperparameters['batch_size']):(j+1)*(self.arch_hyperparameters['batch_size'])]
                    labels_real = labels[j*self.arch_hyperparameters['batch_size']:(j+1)*(self.arch_hyperparameters['batch_size'])]
                
                self.back_prop_noise_formualation(batch_real, labels_real)
        
        return loss_history, acc_history