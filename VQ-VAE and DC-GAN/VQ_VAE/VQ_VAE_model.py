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
class VQ_VAE_tinyImagenet(object):
    def __init__(self, dict_size):
        #self.model = None
        #hyperparameters haven't been tuned: using within some prescribed heuristic level
        self.arch_hyperparameters = {'l2_conv': 0.001,
                                     'l2_emb': 0.001,
                                     'dropout': 0.2,
                                     'lr': 0.0001,
                                     'batch_size': 128
                                     }

        #data dimension will be automatically taken care by keras
        self.data_dims = (64, 64, 3)
        self.encode_dims = (26, 26, 64)

        #the idea is getting 16^25 different latents for the decoder (16 is selected as the as the number of code-vectors because root(200) ~ 16, the dataset has 200 labels)
        self.code_book_dims = (dict_size, 64)

        self.optimizer = None
        self.code_book = None
        self.Encoder = None
        self.Decoder = None
        self.commit_beta = 1    #paper said anything between 0.1-2 has similar effects


    def re_assign_lr(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])
        
    def save_model(self, path):
        self.Encoder.save(path+"Encoder")
        self.Decoder.save(path+"Decoder")
        self.code_book.save(path+"code_book")

    def load_mymodel(self, path):
        print("Loading Model")
        self.Encoder = load_model(path+"Encoder")
        self.Encoder.compile()
        self.Encoder.summary()

        self.code_book = load_model(path+"code_book")
        self.code_book.compile()
        self.code_book.summary()

        self.Decoder = load_model(path+"Decoder")
        self.Decoder.compile()
        self.Decoder.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])

    def encoder(self, inp):
        #Encoder
        #Note first 2 elements are number of maps in the op and the kernel size of the layer
        #Layer e1
        enc_op = Conv2D(16, (5,5), strides=(2,2), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'enc_CONV_L1')(inp)      #l1_conv
        #enc_op = MaxPool2D((5, 5), strides=(2,2), padding='valid', name = 'enc_MaxPool_L1')(enc_op)   #l1_maxpool
        #layer e2
        enc_op = Conv2D(32, (3,3), strides=(1,1), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'enc_CONV_L2')(enc_op)   #l2_conv
        #enc_op = MaxPool2D((3, 3), strides=(1,1), padding='valid', name = 'enc_MaxPool_L2')(enc_op)   #l2 maxpool
        #layer e3
        enc_op = Conv2D(64, (3,3), strides=(1,1), padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'enc_CONV_L3')(enc_op)   #l3_conv


        return enc_op
    #this is used to see if the quantizer is giving the output correctly, need not use it
    def quantizer_sanity_check(self, enc_op, quant_val):
        is_correct = True
        emb_val = self.code_book.get_layer("Embedding_latent").get_weights()[0]
        print(emb_val.shape)
        #over examples
        for ex in range(enc_op.shape[0]):
            #over i in (i, j) of feature maps
            for i in range(enc_op.shape[1]):
                for j in range(enc_op.shape[2]):
                    emb_ij = enc_op[ex, i, j]
                    indx_min = -1
                    dist = np.inf
                    #check over all the emb_val
                    for l in range(emb_val.shape[0]):
                        len_ijl = np.linalg.norm(emb_val[l]-emb_ij, 2)

                        if len_ijl < dist:
                            indx_min = l
                            dist = len_ijl

                    if not np.allclose(quant_val[ex, i, j], emb_val[indx_min]):
                        is_correct=False

        return is_correct

    def quantizer(self, enc_op):
        #note 16 here is a dict size, which (experimented with 3 values)
        enc_op_temp = np.expand_dims(enc_op.numpy(), axis=3)  #covert to (b,5,5,1,64)
        emb_vector  =  np.expand_dims(self.code_book.get_layer("Embedding_latent").get_weights()[0], axis= (0,1,2))  #(1,1,1,16,64)
        #this will give (b,5,5,16,64): for every example, i,j we get a (16, 64) dimensional vector, calculate dist -> left with 16 numbers-> choose maximum

        dist_ij = enc_op_temp - emb_vector
        #calculate the distance for every i,j and every subject
        q_index = np.argmin(np.sum(dist_ij*dist_ij, axis=4), axis=3)
        #access these embedding vectors
        quant_enc_op = self.code_book(q_index)
        return quant_enc_op



    def decoder(self, latent_enc):
        #upsampling fc
        #dec_op = Dense(1600, activation='elu', name='dec_fc')(latent_enc)   #increase dimension of latent
        #dec_op = Dropout(self.arch_hyperparameters['dropout'], name = 'dec_Dropout')(dec_op)
        #inverse flatten
        #dec_op = Reshape((5, 5, 64), name='dec_inv_Flatten')(dec_op)  #reshape

        #inverse l3
        dec_op = Conv2DTranspose(32, (3,3), strides=(1,1), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'dec_inv_CONV_L3')(latent_enc)   #l3_convTranspose
        #inverse l2
        dec_op = Conv2DTranspose(16, (4,4), strides=(1,1), activation='elu', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'dec_inv_CONV_L2')(dec_op)   #l2_convTranspose
        #inverse l1
        dec_op = Conv2DTranspose(3, (4,4), strides=(2,2), activation='sigmoid', padding='valid', kernel_regularizer=l2(self.arch_hyperparameters['l2_conv']), name = 'dec_inv_CONV_L1')(dec_op)      #l1_convTranspose

        return dec_op

    def makeModel(self):
        #make the encoder
        inp = Input(shape=self.data_dims, name='enc_Input')
        self.Encoder = Model(inputs=inp, outputs=self.encoder(inp), name='Encoder')
        self.Encoder.compile()
        self.Encoder.summary()
        #make the decoder
        dec_latent_inp = Input(shape = self.encode_dims, name='dec_latent_Input')
        self.Decoder = Model(inputs=dec_latent_inp, outputs=self.decoder(dec_latent_inp), name='Decoder')
        self.Decoder.compile()
        #make the code-book
        quant_inp = Input(shape=[],name='quant_inp')
        self.code_book = Model(inputs= quant_inp, outputs=Embedding(self.code_book_dims[0], self.code_book_dims[1], embeddings_regularizer=l2(self.arch_hyperparameters['l2_emb']), name='Embedding_latent')(quant_inp), name='code_book')
        self.code_book.compile()
        self.code_book.summary()
        #self.model = Model(inputs=inp, outputs=self.Decoder(self.Encoder(inp)), name='VQ_VAE_Mainak')
        #print(self.code_book.get_layer('Embedding_latent').name, self.code_book.get_layer('Embedding_latent').get_weights()[0])
        self.Decoder.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arch_hyperparameters['lr'])


    def forward_prop(self, input_batch):
        #encoder output
        enc_op = self.Encoder(input_batch)
        #print(enc_op.shape)
        #quantize output
        quant_enc_op = self.quantizer(enc_op)
        #print(quant_enc_op.shape)

        #this was written to check if the quantization was correct
        #print("Quantization is ok: ",self.quantizer_sanity_check(enc_op.numpy(), quant_enc_op.numpy()))

        #decoder output
        dec_op = self.Decoder(quant_enc_op)
        #print(dec_op.shape)
        return enc_op, quant_enc_op, dec_op

    
    def MSE_calc(self, x, y, rd=[1, 2, 3]):
        return tf.reduce_mean(tf.reduce_sum(((x - y)*(x - y))/2, axis=rd))
        
    def losses(self, inp, dec_op, enc_op, quant_enc_op):
        #log P(x|z_e(x)), given encoder the decoder is thought to output the x, gaussian assumption -> mse
        
        recons_loss = self.MSE_calc(inp, dec_op)
        
        #this is the vq_loss, we need to stop the gradients at encoder output z_e(x), to learn the tables
        vq_loss = self.MSE_calc(tf.stop_gradient(enc_op), quant_enc_op)
        #this is the commitment loss, used to make the encoder commit to the q_values and stay near them
        commit_loss = self.MSE_calc(enc_op, tf.stop_gradient(quant_enc_op))

        return recons_loss, vq_loss, self.commit_beta*commit_loss

    #def back_prop(self, inp, dec_op, enc_op, quant_enc_op):
    def back_prop(self, inp):

        with tf.GradientTape(persistent=True) as tape:
            #enc_op, quant_enc_op, dec_op = self.forward_prop(inp)
            enc_op = self.Encoder(inp)

            quant_enc_op = self.quantizer(enc_op)
            vq_loss = self.MSE_calc(tf.stop_gradient(enc_op), quant_enc_op)
            commit_loss = self.commit_beta*self.MSE_calc(enc_op, tf.stop_gradient(quant_enc_op))

            quant_enc_op = enc_op + tf.stop_gradient(quant_enc_op - enc_op)

            dec_op = self.Decoder(quant_enc_op)
            recons_loss = self.MSE_calc(inp, dec_op)

        #backprop
        #decoder loss depends only on reconstruction loss
        grad_Decoder = tape.gradient(recons_loss, self.Decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grad_Decoder, self.Decoder.trainable_variables))
        #print(self.Decoder.trainable_variables)
        #embeddings are learnt using vq_loss
        grad_Embedding = tape.gradient(vq_loss, self.code_book.trainable_variables)

        self.optimizer.apply_gradients(zip(grad_Embedding, self.code_book.trainable_variables))
        #Encoder has 2 part: 1) from the commitment losss
        grad_Encoding_com_grad = tape.gradient(commit_loss, self.Encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grad_Encoding_com_grad, self.Encoder.trainable_variables))
        #part 2:) reconstruction loss: gradients are copied
        grad_Encoding_recon_loss = tape.gradient(recons_loss, self.Encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grad_Encoding_recon_loss, self.Encoder.trainable_variables))

    def train_Model(self, data, n_epochs):
        #self.Encoder.compile(optimizer=tf.keras.Optimizer.Adam(learning_rate=self.arch_hyperparameters['lr']))
        #self.code_book.compile(optimizer=tf.keras.Optimizer.Adam(learning_rate=self.arch_hyperparameters['lr']))
        #self.Decoder.compile(optimizer=tf.keras.Optimizer.Adam(learning_rate=self.arch_hyperparameters['lr']))
        ep_length = int(np.ceil(data.shape[0]/self.arch_hyperparameters['batch_size']))
        loss_history = {'recon_loss': [],
                    'vq_loss': [],
                    'comm_loss': []}
        for i in range(n_epochs):
            d_loss = data[np.random.choice(100000, 5000, replace=False)]
            enc_op, quant_enc_op, dec_op = self.forward_prop(d_loss)
            r, v, c = self.losses(d_loss, dec_op, enc_op, quant_enc_op)

            loss_history['recon_loss'].append(r)
            loss_history['vq_loss'].append(v)
            loss_history['comm_loss'].append(c)
            tot_loss = r+v+c
            if i%2 == 0:
                print("Loss at epoch {}: {}".format(i, tot_loss))

            np.random.shuffle(data)
            for j in range(ep_length):
                batch = None
                if j == ep_length -1:
                    batch = data[j*self.arch_hyperparameters['batch_size']:data.shape[0]]
                else:
                    batch = data[j*(self.arch_hyperparameters['batch_size']):(j+1)*(self.arch_hyperparameters['batch_size'])]
                
                #train with the batch
                #self.back_prop(batch, dec_op, enc_op, quant_enc_op)
                self.back_prop(batch)

        return loss_history
