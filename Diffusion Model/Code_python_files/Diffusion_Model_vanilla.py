#import relevant libraries
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Reshape, Conv2DTranspose, Embedding, Concatenate, Multiply, Add, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
import os


class UNET(object):
    def __init__(self, n_layers, n_filters, inp_shape, time_emb_required, freq=1000):
        self.arch_hyperparameters = {'l2_conv': 0.0001
                                }
        self.inp_shape = inp_shape
        self.time_emb_required = time_emb_required
        #for using the same u-net code with and without using an embedding
        if self.time_emb_required == True:
            #self.omega_emb = np.random.rand(1, self.inp_shape[0], 1, 1)
            self.omega_emb = tf.constant(np.ones((1, inp_shape[0], inp_shape[1], 1))*2*np.pi/(freq+1), dtype='float32', name='omega_emb')
            #form the shape of a input feature map
            #self.omega_emb = tf.constant(np.repeat(self.omega_emb, self.inp_shape[1], axis=2), dtype='float32', name='omega_emb')
            #the phase is selected randomly and uniformly from [0, 2pi], just using 1 vector along the columns
            #self.phi_emb = np.random.rand(1, self.inp_shape[0], 1, 1)*2*np.pi
            #self.phi_emb = np.random.rand(1, 1, 1, 1)*2*np.pi
            #self.phi_emb = tf.constant(np.repeat(self.phi_emb, self.inp_shape[1], axis=2), dtype='float32', name='phi_emb')
            self.phi_emb = np.array([[np.zeros((inp_shape[1],1)), np.ones((inp_shape[1],1))*np.pi/2]])
            self.phi_emb = np.repeat(self.phi_emb, int(inp_shape[0]/2), 1)
            self.phi_emb = tf.constant(self.phi_emb, dtype='float32', name='phi_emb')
            #print(self.omega_emb.shape, self.phi_emb.shape)
        
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.unet_model = None
    

    #before you begin idea is to add an embedding channel to make the data (None, m, n, 4), to propagate this information,
    #a linear 1x1 convolution is done in the image, followed by a (None, m, n, 3)
    def get_sinusoidal_embedding_channel(self, t, w_inp, phi_inp):
        #t_i(j) = sin(w_j t  + phi_j) : i is the time of input
        m = Multiply(name = "wt")([w_inp, t])
        a = Add(name='wt_plus_phi')([m, phi_inp])

        t_emb = Lambda(lambda x: tf.math.sin(x), name='final_time_embedding')(a)

        return t_emb

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
        t_emb = None
        w = None
        phi = None
        #get time embedding
        if self.time_emb_required == True:
            w = Input(shape=self.omega_emb.shape[1:], name='freq_consts')
            phi = Input(shape=self.omega_emb.shape[1:], name='phase_consts')
            
            t = Input(shape=[], name='time_embedding')
            t2 = Reshape(target_shape=(1,1,1), name='reshape_time')(t)
            t_emb = self.get_sinusoidal_embedding_channel(t2, w, phi)
            for_dat = Concatenate(name="input_time_added")([inp, t_emb])

            for_dat = Conv2D(self.n_filters[0], (1,1), strides=(1,1), activation=None, padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'inp_mix_time_info')(for_dat)
        #for embedding free network
        else:
            for_dat = inp

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

        #final layer to convert to (-1, 1)
        if self.time_emb_required == True:
            for_dat = Concatenate(name="output_time_added")([for_dat, t_emb])
            for_dat = Conv2D(self.n_filters[0], (1,1), strides=(1,1), activation='tanh', padding='same', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'scale_output_final')(for_dat)
            self.unet_model = Model(inputs=[inp, t, w, phi], outputs=for_dat, name='Unet_model')
        else:
            self.unet_model = Model(inputs=inp, outputs=for_dat, name='Unet_model')

        self.unet_model.compile()
        #self.unet_model.summary()

class Diffusion_Model(object):
    def __init__(self, input_shape, T):
        #T is the mixing time
        self.input_shape = input_shape
        self.arch_hyperparameters = {
                                     'sub_sample_t': 10,
                                     'batch_size': 16,
                                     'lr': 0.001,
                                     'T': T,
                                     'alpha_t': np.linspace((1-1e-4), 0.97, T+1),
                                     'alpha_t_bar': [],
                                     'sigma_q_t_2': [-1, -1]
                                    }
        self.calculate_alphas()
        
        self.omega_emb = None
        self.phi_emb = None
        
        self.u_net_diff = None
        self.optimizer = None
        
    
    def make_model(self):
        #u_net_recons = UNET(4, [3, 16, 32, 64], self.input_shape, False, self.arch_hyperparameters['T'])
        #u_net_recons.make_UNET()
        #self.u_net_recons = u_net_recons.unet_model
        
        u_net_diff = UNET(5, [3, 32, 64, 96, 128], self.input_shape, True, self.arch_hyperparameters['T'])
        u_net_diff.make_UNET()
        #assign the models
        self.u_net_diff = u_net_diff.unet_model
        
        self.u_net_diff.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])
        
        self.omega_emb = u_net_diff.omega_emb
        self.phi_emb = u_net_diff.phi_emb
        
    def save_model(self, path):
        self.u_net_diff.save(path+"unet_diff")
        #self.u_net_recons.save(path+"unet_recons")
        #save the embeddings
        with open(path+"omega_emb.npy", 'wb') as fptr:
            np.save(fptr, self.omega_emb)
        with open(path+"phi_emb.npy", 'wb') as fptr:
           np.save(fptr, self.phi_emb)
        
    def load_mymodel(self, path):
        print("Loading Model")
        self.u_net_diff = load_model(path+"unet_diff")
        self.u_net_diff.compile()
        self.u_net_diff.summary()
    
        #self.u_net_recons = load_model(path+"unet_recons")
        #self.u_net_recons.compile()
        #self.u_net_recons.summary()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])
        #load the embeddings
        self.omega_emb = np.load(path+"omega_emb.npy")
        self.omega_emb = tf.constant(self.omega_emb, dtype='float32', name='omega_emb_val')
        self.phi_emb = np.load(path+"phi_emb.npy")
        self.phi_emb = tf.constant(self.phi_emb, dtype='float32', name='omega_emb_val')
        
    def calculate_alphas(self):
        #using dyanamic programming approach to keep constants stored to prevent recalculation
        #a_0 = self.arch_hyperparameters['alpha_0']
        for i in range(self.arch_hyperparameters['T']+1):
            #schedule: a_t = a_0/(i+1)
            #a_t = a_0/(i+1)
            #self.arch_hyperparameters['alpha_t'].append(a_t)
            #cumulative product a_t_bar
            if i <= 1:
                self.arch_hyperparameters['alpha_t_bar'].append(self.arch_hyperparameters['alpha_t'][i])
            else:
                a_t_1_bar = self.arch_hyperparameters['alpha_t_bar'][i-1]
                self.arch_hyperparameters['alpha_t_bar'].append(a_t_1_bar*self.arch_hyperparameters['alpha_t'][i])
            
            #sigma_q_t_2 is for t>=2
            if i>=2:
                sqt2 = (1-self.arch_hyperparameters['alpha_t'][i])*(1-self.arch_hyperparameters['alpha_t_bar'][i-1])/(1-self.arch_hyperparameters['alpha_t_bar'][i])
                self.arch_hyperparameters['sigma_q_t_2'].append(sqt2)
        
        #self.arch_hyperparameters['alpha_t'] = np.array(self.arch_hyperparameters['alpha_t'])
        self.arch_hyperparameters['alpha_t_bar'] = np.array(self.arch_hyperparameters['alpha_t_bar'])
        self.arch_hyperparameters['sigma_q_t_2'] = np.array(self.arch_hyperparameters['sigma_q_t_2'])
        print(self.arch_hyperparameters['sigma_q_t_2'])
        
    def X_t(self, data_X_0, t_vector):
        #returns the noise added images
        u_xt_giv_x0 = np.expand_dims(np.sqrt(self.arch_hyperparameters['alpha_t_bar'][t_vector]), axis=[1,2,3])*data_X_0
        sigma_xt_giv_x0 = np.expand_dims(np.sqrt((1 - self.arch_hyperparameters['alpha_t_bar'][t_vector])), axis=[1,2,3])
        
        eps = np.random.normal(size=data_X_0.shape) 
        
        X_t = u_xt_giv_x0 + sigma_xt_giv_x0*eps
        return X_t, eps
    
    def loss(self, batch):
        #t1s = np.ones(batch.shape[0], dtype='int32')
        #batch_X1s = self.X_t(batch, t1s)
        
        #get all X_ts random selected for each image
        #generate a list of random numbers [2, T]*num_image*number_estimate for each, 1 is the hack to learn the last step too
        ts = np.random.randint(1, self.arch_hyperparameters['T']+1, size=self.arch_hyperparameters['sub_sample_t']*batch.shape[0])
        
        #repeat the first dimension t times (to estimate the sum 2 to T)
        batch_appended = np.repeat(batch, self.arch_hyperparameters['sub_sample_t'], axis=0)
        
        batch_appended, noise = self.X_t(batch_appended, ts)

        #X_0_recon = self.u_net_recons(batch_X1s)
        
        ep_theta_ts = self.u_net_diff([batch_appended, ts, self.omega_emb, self.phi_emb])
        
        #noise = np.random.normal(size=batch_appended.shape)
            
        #reconstruction loss (we maximizing negative of mse)
        #mse_recon = tf.reduce_sum(((X_0_recon - batch)**2)/2, axis=[1, 2, 3])
        #loss_recon = tf.reduce_mean(mse_recon)
            
        #averaging per sample only, diffusion loss
        #weights = ((1-self.arch_hyperparameters['alpha_t'][ts])**2)/(self.arch_hyperparameters['sigma_q_t_2'][ts]*(1-self.arch_hyperparameters['alpha_t_bar'][ts])*self.arch_hyperparameters['alpha_t'][ts])
        #weights = np.expand_dims(weights, axis=[1,2,3])
        
        #MSE loss
        #mse_diff = tf.reduce_sum(((ep_theta_ts - noise)**2)/2, axis=[1, 2, 3])
        #loss_diff = tf.reduce_mean(mse_diff)/self.arch_hyperparameters['sub_sample_t']
        #MAE loss
        mae_diff = tf.reduce_sum(tf.math.abs(ep_theta_ts - noise), axis=[1, 2, 3])
        loss_diff = tf.reduce_mean(mae_diff)/self.arch_hyperparameters['sub_sample_t']
        del(batch_appended)
        
        #return loss_recon, loss_diff
        return loss_diff
        
    def back_prop_noise_formualation(self, batch):
        #get all the X1s
        #t1s = np.ones(batch.shape[0], dtype='int32')
        
        #batch_X1s = self.X_t(batch, t1s)
        #get all X_ts random selected for each image
        #generate a list of random numbers [1, T]*num_image*number_estimate for each
        ts = np.random.randint(1, self.arch_hyperparameters['T']+1, size=self.arch_hyperparameters['sub_sample_t']*batch.shape[0])
        
        #repeat the first dimension t times (to estimate the sum 2 to T)
        batch_appended = np.repeat(batch, self.arch_hyperparameters['sub_sample_t'], axis=0)
        batch_appended, noise = self.X_t(batch_appended, ts)
        
        #I am minimizing the negative of the ELBO
        #train the model
        with tf.GradientTape(persistent=True) as tape:
            #X_0_recon = self.u_net_recons(batch_X1s)
            ep_theta_ts = self.u_net_diff([batch_appended, ts, self.omega_emb, self.phi_emb])
            #noise = np.random.normal(size=batch_appended.shape)
            
            #reconstruction loss
            #mse_recon = tf.reduce_sum(((X_0_recon - batch)**2)/2, axis=[1, 2, 3])
            #loss_recon = tf.reduce_mean(mse_recon)
            
            #averaging per sample only, diffusion loss
            #weights = ((1-self.arch_hyperparameters['alpha_t'][ts])**2)/(self.arch_hyperparameters['sigma_q_t_2'][ts]*(1-self.arch_hyperparameters['alpha_t_bar'][ts])*self.arch_hyperparameters['alpha_t'][ts])
            #weights = np.expand_dims(weights, axis=[1,2,3])
            #MSE loss
            #mse_diff = tf.reduce_sum(((ep_theta_ts - noise)**2)/2, axis=[1, 2, 3])
            #loss_diff = tf.reduce_mean(mse_diff)/self.arch_hyperparameters['sub_sample_t']
            #MAE loss
            mae_diff = tf.reduce_sum(tf.math.abs(ep_theta_ts - noise), axis=[1, 2, 3])
            loss_diff = tf.reduce_mean(mae_diff)/self.arch_hyperparameters['sub_sample_t']
        
        #calculate gradient
        #grad_recon = tape.gradient(loss_recon, self.u_net_recons.trainable_variables)
        grad_diff = tape.gradient(loss_diff, self.u_net_diff.trainable_variables)
        
        #gradient descent
        #self.optimizer.apply_gradients(zip(grad_recon, self.u_net_recons.trainable_variables))
        self.optimizer.apply_gradients(zip(grad_diff, self.u_net_diff.trainable_variables))
        
        del(batch_appended)
    
    def prep_image(self, batch):
        #rescale to [-1, 1], as we will use batch norms later
        
        mx = np.max(batch, axis = (1,2,3), keepdims=True)
        mn = np.min(batch, axis = (1,2,3), keepdims=True)
        batch = batch.astype('float32')
        batch = 2*(batch-mn)/(mx-mn) - 1
        
        return batch
    
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
                app = tf.clip_by_value((255*(X_t+1)/2), 0, 255)
                sampled_langevian_steps.append(app.numpy().astype('uint8'))
            
            X_t = tf.clip_by_value(X_t, -1, 1)
            
            z = 0
            if t>1:
                z = np.random.normal(size=(num_img, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            
            ts = t*np.ones(num_img, dtype='float32')
            eps_theta_X_t = self.u_net_diff([X_t, ts, self.omega_emb, self.phi_emb])
            
            w1 = 1/np.sqrt(self.arch_hyperparameters['alpha_t'][t])
            
            w2 = (1-self.arch_hyperparameters['alpha_t'][t])/(np.sqrt((1-self.arch_hyperparameters['alpha_t_bar'][t])*self.arch_hyperparameters['alpha_t'][t]))
            w3 = 0
            if t>1:
                w3 = np.sqrt(self.arch_hyperparameters['sigma_q_t_2'][t])
            
            X_t = w1*X_t - w2*eps_theta_X_t + w3*z 
            
            
        
        #final image
        app = tf.clip_by_value((255*(X_t+1)/2), 0, 255)
        sampled_langevian_steps.append(app.numpy().astype('uint8'))
        
        return sampled_langevian_steps
    
    def train_diff_model_noise_formulation(self, data, n_epochs):
        
        ep_length = int(data.shape[0]/self.arch_hyperparameters['batch_size'])
        #loss_history = {
        #                'recons': [],
        #                'diff': []
        #                }
        
        loss = []
        for i in range(n_epochs):
            samples = self.prep_image(data[np.random.choice(data.shape[0], self.arch_hyperparameters['sub_sample_t'], replace=False)])
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
                
                batch = data[j*(self.arch_hyperparameters['batch_size']):(j+1)*(self.arch_hyperparameters['batch_size'])]
            
                self.back_prop_noise_formualation(self.prep_image(batch))
        
        return loss
