#import relevant libraries
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Reshape, Conv2DTranspose, Embedding
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
import os

#VQ-VAE architechture
class VAE_vanilla(object):
    def __init__(self):
        #self.model = None
        #hyperparameters haven't been tuned: using within some prescribed heuristic level
        self.arch_hyperparameters = {'l2_conv': 0.001,
                                     'l2_emb': 0.001,
                                     'dropout': 0.2,
                                     'lr': 0.00015,
                                     'batch_size': 16
                                     }

        #data dimension will be automatically taken care by keras
        self.data_dims = (26, 26, 64)
        self.encode_dims = (10, 10, 64)

        self.optimizer = None
        self.Encoder = None
        self.Decoder = None
        self.beta = 1
    
    def re_assign_lr(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])
        
    def save_model(self, path):
        self.Encoder.save(path+"Encoder")
        self.Decoder.save(path+"Decoder")

    def load_mymodel(self, path):
        print("Loading Model")
        self.Encoder = load_model(path+"Encoder")
        self.Encoder.compile()
        self.Encoder.summary()

        self.Decoder = load_model(path+"Decoder")
        self.Decoder.compile()
        self.Decoder.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])

    def encoder(self, inp):
        #Encoder
        #Layer e1
        enc_op = Conv2D(64, (3,3), strides=(2,2), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'enc_CONV_L1_mu')(inp)      #l1_conv
        #layer e2
        enc_op_mu = Conv2D(64, (3,3), strides=(1,1), padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'enc_CONV_L2_mu')(enc_op)   #l2_conv

        #just a regression layer, standard deviation is always positive: make it encode log (std)
        enc_op_sigma = Conv2D(64, (3,3), strides=(1,1), padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'enc_CONV_L2_sigma')(enc_op)   #l2_conv


        enc_op_mu = Flatten(name='Flatten_mu_enc')(enc_op_mu)
        enc_op_sigma = Flatten(name='Flatten_sigma_enc')(enc_op_sigma)
        return enc_op_mu, enc_op_sigma

    def reparameterization(self, enc_mu, enc_sigma):
        #here we are doing a 1-1 reparameterization
        norm_0_1 = tf.random.normal(shape=enc_sigma.shape)
        #need to take exponents of variance in order to invert the log that we modeled in the NN
        z = enc_mu + tf.exp(enc_sigma)*norm_0_1
        return z

    def decoder(self, latent_enc):

        #inverse l2
        latent_enc = Reshape(self.encode_dims, name="Reshape_dec")(latent_enc)
        dec_op = Conv2DTranspose(64, (3,3), strides=(1,1), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'dec_inv_CONV_L3')(latent_enc)   #l3_convTranspose
        #inverse l1
        dec_op = Conv2DTranspose(64, (4,4), strides=(2,2), padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'dec_inv_CONV_L1')(dec_op)      #l1_convTranspose

        return dec_op

    def makeModel(self):
        #make the encoder
        inp = Input(shape=self.data_dims, name='enc_Input')
        self.Encoder = Model(inputs=inp, outputs=self.encoder(inp), name='Encoder')
        self.Encoder.compile()
        self.Encoder.summary()
        #make the decoder
        dec_latent_inp = Input(shape = self.encode_dims[0]*self.encode_dims[1]*self.encode_dims[2], name='dec_latent_Input')
        self.Decoder = Model(inputs=dec_latent_inp, outputs=self.decoder(dec_latent_inp), name='Decoder')

        self.Decoder.compile()
        self.Decoder.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])


    def forward_prop(self, input_batch):
        #encoder output
        enc_op_mu, enc_op_sigma = self.Encoder(input_batch)
        #print(enc_op_mu.shape, enc_op_sigma.shape)
        reparam_op = self.reparameterization(enc_op_mu, enc_op_sigma)
        #print(reparam_op.shape)
        #decoder output
        dec_op = self.Decoder(reparam_op)
        #print(dec_op.shape)
        return enc_op_mu, enc_op_sigma, reparam_op, dec_op

    def log_Normal(self, X, mu, sigma):
        #considering indepenent dimensions
        v = tf.reduce_sum(((X - mu)*(X - mu))/(2*sigma*sigma) + tf.math.log(sigma), axis=1)
        #print(v.shape)
        return -1*v

    def losses(self, inp, sample_q_z_given_x, mean_x, std_x, dec_op):
        #log P(x|z), given encoder the decoder is thought to output the x, gaussian assumption -> mse
        recons_loss = tf.reduce_sum(((inp - dec_op)*(inp - dec_op))/2, axis=[1, 2, 3])
        #The KL between the q_z_x, and
        #computing the sample wise k-l loss E(log q(z/x)/p(z))
        KL_loss = tf.reduce_mean(self.log_Normal(sample_q_z_given_x, mean_x, tf.exp(std_x)) - self.log_Normal(sample_q_z_given_x, 0., 1.))

        return tf.reduce_mean(recons_loss), self.beta*KL_loss

    #def back_prop(self, inp, dec_op, enc_op, quant_enc_op):
    def back_prop(self, inp):

        with tf.GradientTape(persistent=True) as tape:
            #enc_op, quant_enc_op, dec_op = self.forward_prop(inp)
            enc_op_mu, enc_op_sigma = self.Encoder(inp)

            reparam_enc_op = self.reparameterization(enc_op_mu, enc_op_sigma)
            dec_op = self.Decoder(reparam_enc_op)

            recons_loss = tf.reduce_sum(((inp - dec_op)*(inp - dec_op))/2, axis=[1, 2, 3])
            KL_loss = tf.reduce_mean(self.log_Normal(reparam_enc_op, enc_op_mu, tf.exp(enc_op_sigma)) - self.log_Normal(reparam_enc_op, 0., 1.))

            loss = tf.reduce_mean(recons_loss) + self.beta*KL_loss

        #backprop
        #decoder loss depends only on reconstruction loss
        grad_Decoder = tape.gradient(loss, self.Decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grad_Decoder, self.Decoder.trainable_variables))

        #part 2:) reconstruction loss: gradients are copied
        grad_Encoder = tape.gradient(loss, self.Encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grad_Encoder, self.Encoder.trainable_variables))

    def train_Model(self, data, n_epochs):
        #self.Encoder.compile(optimizer=tf.keras.Optimizer.Adam(learning_rate=self.arch_hyperparameters['lr']))
        #self.code_book.compile(optimizer=tf.keras.Optimizer.Adam(learning_rate=self.arch_hyperparameters['lr']))
        #self.Decoder.compile(optimizer=tf.keras.Optimizer.Adam(learning_rate=self.arch_hyperparameters['lr']))
        ep_length = int(np.ceil(data.shape[0]/self.arch_hyperparameters['batch_size']))
        loss_history = {'mse_loss': [],
                        'KL_loss': []
                       }
        for i in range(n_epochs):
            d_loss = data[np.random.choice(data.shape[0], int(0.01*data.shape[0]), replace=False)]

            enc_op_mu, enc_op_sigma, reparam_op, dec_op = self.forward_prop(d_loss)

            reparam_op = self.reparameterization(enc_op_mu, enc_op_sigma)
            dec_op = self.Decoder(reparam_op)
            
            m, k = self.losses(d_loss, reparam_op, enc_op_mu, enc_op_sigma, dec_op)

            loss_history['mse_loss'].append(m)
            loss_history['KL_loss'].append(k)
            tot_loss = m+k
            if i%2 == 0:
                print("Loss at epoch {}: {}".format(i, tot_loss))

            np.random.shuffle(data)
            for j in range(ep_length):
                batch = None
                if j == ep_length -1:
                    batch = data[j*self.arch_hyperparameters['batch_size']:data.shape[0]]
                else:
                    batch = data[j*(self.arch_hyperparameters['batch_size']):(j+1)*(self.arch_hyperparameters['batch_size'])]
                
                self.back_prop(batch)

        return loss_history

    def sample_VAE(self, n_samples):
        sample_norm = tf.random.normal(shape=(n_samples, self.encode_dims[0]*self.encode_dims[1]*self.encode_dims[2]))

        return self.Decoder(sample_norm)

