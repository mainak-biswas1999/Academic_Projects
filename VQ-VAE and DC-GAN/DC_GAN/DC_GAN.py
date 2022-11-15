import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout, Reshape, Conv2DTranspose, Embedding 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os

class DC_GAN(object):
    def __init__(self):
        #These are suggested by DC gan paper
        self.arch_hyperparameters = {'l2_conv': 0.0001, 
                                     'dropout': 0.2,
                                     'lr': 0.0002,
                                     'beta1': 0.5,
                                     'batch_size': 128
                                     }
        
        self.noise_input= 100
        self.gen_inp_dim = (8, 8, 256)
        self.data_dims = (128, 128, 3)
        self.Generator = None
        self.Discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None

    def save_model(self, path):
        self.Generator.save(path+"Generator")
        self.Discriminator.save(path+"Discriminator")
    
    def load_mymodel(self, path):
        print("Loading Model")
        self.Generator = load_model(path+"Generator")
        self.Generator.compile()
        self.Generator.summary()
    
        self.Discriminator = load_model(path+"Discriminator")
        self.Discriminator.compile()
        self.Discriminator.summary()
        
        self.optimizer_g = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'], beta_1=self.arch_hyperparameters['beta1'])   
        self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'], beta_1=self.arch_hyperparameters['beta1'])
    def generator(self, inp):
        #note: to use batch norm the data must have 0 mean atleast (pass image only after rescaling to [-1, 1])
        
        gen_op = Dense(self.gen_inp_dim[0]*self.gen_inp_dim[1]*self.gen_inp_dim[2], use_bias=False, name='project_enc')(inp)
        gen_op = BatchNormalization(name='gen_batch_norm_project')(gen_op)
        gen_op = Reshape(self.gen_inp_dim, name='reshape_enc')(gen_op)

        #the dc-gan tells us to use convolutions and batch norms and leaky relu (I am using elu, which is seen to be better)
        #inv_conv layer 1
        gen_op = Conv2DTranspose(128, (5, 5), strides=(2,2), padding='same', activation='relu', use_bias=False, name='gen_inv_conv_l1')(gen_op)
        #paper said use batch norm instead of regularization, normalization was done the channel dimension 
        gen_op = BatchNormalization(name='gen_batch_norm_l1')(gen_op)
        gen_op = Conv2DTranspose(64, (5, 5), strides=(2,2), padding='same', activation='relu', use_bias=False, name='gen_inv_conv_l2')(gen_op)
        gen_op = BatchNormalization(name='gen_batch_norm_l2')(gen_op)
        gen_op = Conv2DTranspose(24, (5, 5), strides=(2,2), padding='same', activation='relu', use_bias=False, name='gen_inv_conv_l3')(gen_op)
        gen_op = BatchNormalization(name='gen_batch_norm_l3')(gen_op)
        #we want the output [-1, 1]: if you are using a batch normalization
        gen_op = Conv2DTranspose(3, (5, 5), strides=(2,2), padding='same', activation='tanh', use_bias=False, name='gen_inv_conv_l4')(gen_op)
        
        return gen_op
    
    def discriminator(self, inp):
        #conv_layer 1
        disc_op = Conv2D(12, (5,5), strides=(3,3), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'disc_conv_l1')(inp)
        #conv layer 2
        disc_op = Conv2D(24, (5,5), strides=(3,3), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'disc_conv_l2')(disc_op)
        disc_op = BatchNormalization(name='disc_batch_norm_l2')(disc_op)
        #conv layer 3
        disc_op = Conv2D(36, (3,3), strides=(2,2), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'disc_conv_l3')(disc_op)
        disc_op = BatchNormalization(name='disc_batch_norm_l3')(disc_op)
        #Flatten
        disc_op = Flatten(name='disc_flatten')(disc_op)
        #disc_op = Dropout(self.arch_hyperparameters['dropout'], name='disc_dropout')(disc_op)
        
        disc_op = Dense(1, activation='sigmoid', name='disc_output_layer')(disc_op)
        
        return disc_op
    
    def makeModel(self):
        #noise inp
        n_inp = Input(shape=self.noise_input, name='gen_input')
        self.Generator = Model(inputs=n_inp, outputs=self.generator(n_inp), name='Generator')
        self.Generator.compile()
        self.Generator.summary()
        #discriminator 
        disc_inp = Input(shape=self.data_dims, name='disc_input')
        self.Discriminator = Model(inputs=disc_inp, outputs=self.discriminator(disc_inp), name='Discriminator')
        self.Discriminator.compile()
        self.Discriminator.summary()
        
        self.optimizer_g = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'], beta_1=self.arch_hyperparameters['beta1'])
        self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'], beta_1=self.arch_hyperparameters['beta1'])
   
    def generate_image(self, bs):
        noise_vol = tf.random.uniform(minval=-1, maxval=1, shape=(bs, self.noise_input))
        return self.Generator(noise_vol)
    
    def loss(self, data_real, data_gen, JS_flag):
        rdat_log_likelihood = tf.reduce_mean(tf.math.log(self.Discriminator(data_real)))
        fake_log_likelihood = tf.reduce_mean(tf.math.log(1-self.Discriminator(data_gen)))
        
        gen_loss = None
        disc_loss = -1*(rdat_log_likelihood + fake_log_likelihood)
        if JS_flag == 1:
            #JS divergence
            gen_loss = fake_log_likelihood
        else:
            #this is the improvised loss
            gen_loss = tf.reduce_mean(-1*tf.math.log(self.Discriminator(data_gen)))
        return gen_loss, disc_loss
    
    def get_fake_real_acc(self, real, fake_data):
        r_acc = np.sum(self.Discriminator(real).numpy() >= 0.5)/real.shape[0]
        f_acc = np.sum(self.Discriminator(fake_data).numpy() < 0.5)/fake_data.shape[0]
        return r_acc, f_acc
    
    def back_prop_JS(self, data_real, data_gen_size, train_disc_flag=False):
        
        with tf.GradientTape(persistent=True) as tape:
            rdat_log_likelihood = None
            
            data_gen = self.generate_image(data_gen_size)

            if train_disc_flag == True:
                rdat_log_likelihood = tf.reduce_mean(tf.math.log(self.Discriminator(data_real)))
            #JS divergence 
            fake_log_likelihood = tf.reduce_mean(tf.math.log(1-self.Discriminator(data_gen)))
            #this is improvised loss
            #fake_log_likelihood = tf.reduce_mean(-1*tf.math.log(self.Discriminator(data_gen)))
            gen_loss = fake_log_likelihood 
            if train_disc_flag == True:
              #we try to maximize the sum and hence taking the negative
              disc_loss = -1*(fake_log_likelihood + rdat_log_likelihood)
        
        #train Generator
        gen_grad = tape.gradient(gen_loss, self.Generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(gen_grad, self.Generator.trainable_variables))
        #train Discriminator
        if train_disc_flag==True:
            disc_grad = tape.gradient(disc_loss, self.Discriminator.trainable_variables)
            self.optimizer_d.apply_gradients(zip(disc_grad, self.Discriminator.trainable_variables))
    
    def back_prop_impro(self, data_real, data_gen_size, disc_cycle):
        #train gen n times
        #noise = tf.random.uniform(minval=-1, maxval=1, shape=(data_gen_size, self.noise_input))
        for i in range(disc_cycle):
            with tf.GradientTape(persistent=True) as tape:
                #data_gen = self.Generator(noise)
                data_gen = self.generate_image(data_gen_size)
            
                rdat_log_likelihood = tf.reduce_mean(tf.math.log(self.Discriminator(data_real)))
                #this is improvised loss
                fake_log_likelihood = tf.reduce_mean(tf.math.log(1-self.Discriminator(data_gen)))
                
                #we try to maximize the sum and hence taking the negative
                disc_loss = -1*(fake_log_likelihood + rdat_log_likelihood)
            #train Discriminator
            disc_grad = tape.gradient(disc_loss, self.Discriminator.trainable_variables)
            self.optimizer_d.apply_gradients(zip(disc_grad, self.Discriminator.trainable_variables))
        
        with tf.GradientTape(persistent=True) as tape:
            data_gen = self.generate_image(data_gen_size)
            gen_loss = tf.reduce_mean(-1*tf.math.log(self.Discriminator(data_gen)))
        
        #train Generator
        gen_grad = tape.gradient(gen_loss, self.Generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(gen_grad, self.Generator.trainable_variables))

    
    def GAN_generate(self, batch_sz):
        batch = self.generate_image(batch_sz).numpy()
        #get the image scale back
        return (255*(batch+1)/2).astype('uint8')
    
    def prep_image(self, batch):
        #rescale to [-1, 1], as we will use batch norms later
        
        mx = np.max(batch, axis = (1,2,3), keepdims=True)
        mn = np.min(batch, axis = (1,2,3), keepdims=True)
        batch = batch.astype('float32')
        batch = 2*(batch-mn)/(mx-mn) - 1
        
        return batch
    
    def train_GAN(self, data, n_epochs, gen_cycle, JS_flag):
        
        ep_length = int(np.ceil(data.shape[0]/self.arch_hyperparameters['batch_size']))
        loss_history = {'gen_loss': [],
                        'disc_loss': []
                       }
        acc_history = {'r_acc': [],
                       'f_acc': [], 
                        }
        for i in range(n_epochs):
            d_loss_real = self.prep_image(data[np.random.choice(data.shape[0], 1000, replace=False)])
            d_loss_gen = self.generate_image(1000)
            
            gl, dl = self.loss(d_loss_real, d_loss_gen, JS_flag)
            r_acc, f_acc = self.get_fake_real_acc(d_loss_real, d_loss_gen)
            
            loss_history['disc_loss'].append(dl)
            loss_history['gen_loss'].append(gl)
            
            acc_history['r_acc'].append(r_acc)
            acc_history['f_acc'].append(f_acc)
            
            if i%2 == 0:
                print("Losses at epoch {}, generator: {} discriminator: {} \n Real Acc: {} Fake Acc: {}".format(i, gl, dl, r_acc, f_acc))
            
            np.random.shuffle(data)
            for j in range(ep_length):
                batch_real = None
                batch_fake_sz = self.arch_hyperparameters['batch_size']
                #batch_fake = self.generate_image(self.arch_hyperparameters['batch_size'])
                if j == ep_length - 1:
                    batch_real = data[j*self.arch_hyperparameters['batch_size']:data.shape[0]]
                else:
                    batch_real = data[j*(self.arch_hyperparameters['batch_size']):(j+1)*(self.arch_hyperparameters['batch_size'])]
            
                if JS_flag == 1:
                    #original image is between 0, 255, convert to [-1, 1]
                    #train the discriminator once in every gen_cycle examples
                    if j%gen_cycle == 0:
                        self.back_prop_JS(self.prep_image(batch_real), batch_fake_sz, True)
                    else:
                        self.back_prop_JS(self.prep_image(batch_real), batch_fake_sz, False)
                
                else:
                    self.back_prop_impro(self.prep_image(batch_real), batch_fake_sz, gen_cycle)
        return loss_history, acc_history
