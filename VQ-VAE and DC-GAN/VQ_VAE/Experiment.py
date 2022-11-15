from PIL import Image
from VQ_VAE_model import *
from Vanilla_VAEs import *
from GAN import *
def save_grid(data, savePath):
    #tiny image is 64x64 
    grid_10_10 = Image.new('RGB', (640,640))
    idx = 0
    for i in range(0, 640, 64):
        for j in range(0, 640, 64):
            
            im = (data[idx]*255).astype('uint8')
            im = Image.fromarray(im)
            grid_10_10.paste(im, (i,j))
            idx += 1
    grid_10_10.save(savePath)
    
def save_10_10(dataset, model, savePath):
    inp_img = dataset[np.random.choice(dataset.shape[0], 100, replace=False)]
    save_grid(inp_img, savePath+"orig.jpg")
    
    _, _, recon_op = model.forward_prop(inp_img)
    #recons_batch = model.VAE_reconstruct(model.prepare_image(inp_img))
    save_grid(recon_op.numpy(), savePath+"recon.jpg")
    

def plot_comp_curves(losses, model_det, image_saveloc):
    plt.plot(losses['recon_loss'], label='recon loss')
    plt.plot(losses['vq_loss'], label='vq loss')
    #plt.plot(losses['recon_loss'] + losses['comm_loss'], label='total loss')
    plt.legend()
    plt.xlabel('Epoch', size=15)
    plt.ylabel('Loss', size=15)
    plt.title('Learning Curve with code book length= '+str(model_det), size=15)
    plt.savefig(image_saveloc+"Learning_curve_"+str(model_det)+".png")
    #plt.show()
    
    
def DriverCode(train_loc, cv_loc, save_loc, image_saveloc,  code_sz):
    #train = train = np.load(train_loc)
    my_model = VQ_VAE_tinyImagenet_new(code_sz)
    #my_model = VQ_VAE_tinyImagenet_new(code_sz)
    if code_sz == 8:
        my_model.arch_hyperparameters['lr'] = 0.0005 
    my_model.makeModel()
    train = np.load(train_loc)
    

    loss_array = my_model.train_Model(train, 8)
    my_model.save_model(save_loc)

    plot_comp_curves(loss_array, code_sz, image_saveloc)
    
    #save grid of reconstruction
    cv_set = train = np.load(cv_loc)
    save_10_10(train, my_model, image_saveloc+"Train_Recon_"+str(code_sz)+"_code")
    save_10_10(cv_set, my_model, image_saveloc+"Test_Recon_"+str(code_sz)+"_code")

    del train
    del cv_set
    del my_model

#GMM
def learn_GMM_on_latents(data, vq_vae_model, latent_loc):
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA 

    #this_model = VQ_VAE_tinyImagenet(32)
    #this_model.load_mymodel(vq_vae_model)
    
    #_, quant_op, _ = this_model.forward_prop(data[np.random.choice(100000, 10000, replace=False)])
    #reshape to (n_exmp, vectorize)
    #quant_op = quant_op.reshape(quant_op.shape[0], -3)
    
    latents = np.load(latent_loc)
    l_shape = latents.shape
    latents = np.reshape(latents, (l_shape[0], l_shape[1]*l_shape[2]*l_shape[3]))
    
    pca_latents = PCA(n_components=26*26)
    pca_latents.fit(latents)
    
    latents_dim_red = pca_latents.transform(latents)
    print(latents_dim_red.shape)
    
    gmm_model = GaussianMixture(128, covariance_type='full', tol=0.001, reg_covar=1e-03, max_iter=50, init_params='kmeans', warm_start=True, verbose=1)
    gmm_model.fit(latents_dim_red)
    
    #for i in range(2):
    #_, quant_op, _ = this_model.forward_prop(data[np.random.choice(100000, 5000, replace=False)])
    #    _, quant_op, _ = this_model.forward_prop(data[i*(5000):(i+1)*5000])
        #reshape to (n_exmp, vectorize)
    #    quant_op = quant_op.numpy()
    #    quant_op = quant_op.reshape(quant_op.shape[0], -3)
    #    gmm_model.fit(quant_op)
    
    return gmm_model, pca_latents

def generate_image(gmm_model, pca_latents, vq_vae_model, reshape_size):
    n = 100
    hundred_samples, gaussians_indx = gmm_model.sample(n_samples=n)    
    hundred_samples = pca_latents.inverse_transform(hundred_samples)
    hundred_samples = np.array(hundred_samples)
    
    hundred_samples = hundred_samples.reshape(n, reshape_size[0], reshape_size[1], reshape_size[2])
    print(hundred_samples.shape)
    
    this_model = VQ_VAE_tinyImagenet(32)
    this_model.load_mymodel(vq_vae_model)
    
    generate_images = this_model.decoder(hundred_samples)
    save_grid(generate_images.numpy(), "Generated_GMMs128comp_VQ_VAE_32code.jpg")


#cv = np.load("./Data/tiny-imagenet-200/cv.npy")
gmm_model, pca_latents = learn_GMM_on_latents(None, "./VQ_VAE_32/model/", "./Latents/latentX.npy")
generate_image(gmm_model, pca_latents, "./VQ_VAE_32/model/", [26, 26, 64])

def plot_comp_curves2(losses, plot_save):
    plt.plot(losses['mse_loss'], label='MSE loss')
    plt.plot(losses['KL_loss'], label='KL loss')
    #plt.plot(losses['recon_loss'] + losses['comm_loss'], label='total loss')
    plt.legend()
    plt.xlabel('Epoch', size=15)
    plt.ylabel('Loss', size=15)
    plt.title('Learning Curve of vanilla VAE on latent')
    plt.savefig(plot_save)

def learn_latents_using_VAE(data, vq_vae_model, code_book_len):  
    
    vq_model = VQ_VAE_tinyImagenet(code_book_len)
    vq_model.load_mymodel(vq_vae_model)

    van_vae = VAE_vanilla()
    van_vae.makeModel()
    data = np.load(data)
    
    
    loss_history = {'mse_loss': [],
                    'KL_loss': []
                    }
    for i in range(1):
        np.random.shuffle(data)
        for j in range(5):
            _, quant_op, _ = vq_model.forward_prop(data[j*5000:(j+1)*5000])
            van_vae.arch_hyperparameters['lr'] /= 1.5
            van_vae.re_assign_lr()

            loss_h = van_vae.train_Model(quant_op.numpy(), 5)
            loss_history['mse_loss'] += loss_h['mse_loss']
            loss_history['KL_loss'] += loss_h['KL_loss']
    
    return van_vae, vq_model, loss_history

def generate_layer_vae(data, vq_vae_model, code_book_len, savepath, save_gen_images):
    van_vae, vq_vae, loss_history = learn_latents_using_VAE(data, vq_vae_model, code_book_len)
    
    plot_comp_curves2(loss_history, savepath)
    
    sample_vae_pt = van_vae.sample_VAE(100)

    gen_images = vq_vae.Decoder(sample_vae_pt)
    save_grid(gen_images.numpy(), save_gen_images)



def learn_latents_using_GAN(data, vq_vae_model, code_book_len):

    vq_model = VQ_VAE_tinyImagenet(code_book_len)
    vq_model.load_mymodel(vq_vae_model)

    GAN_vae = VQ_GAN()
    GAN_vae.makeModel()
    data = np.load(data)


    loss_history = {'gen_loss': [],
                        'disc_loss': []
                       }
    acc_history = {'r_acc': [],
                       'f_acc': [],
                    }
    for i in range(1):
        np.random.shuffle(data)
        for j in range(1):
            _, quant_op, _ = vq_model.forward_prop(data[j*5000:(j+1)*5000])
            #van_vae.arch_hyperparameters['lr'] /= 1.5
            #van_vae.re_assign_lr()

            loss_h, acc_h = GAN_vae.train_GAN(quant_op.numpy(), 5, 3, 0)
            loss_history['gen_loss'] += loss_h['gen_loss']
            loss_history['disc_loss'] += loss_h['disc_loss']
            acc_history['r_acc'] += acc_h['r_acc']
            acc_history['f_acc'] += acc_h['f_acc']

    return GAN_vae, vq_model, loss_history, acc_h

def generate_layer_GAN(data, vq_vae_model, code_book_len, savepath, save_gen_images):
    GAN_vae, vq_vae, loss_history, acc_h = learn_latents_using_VAE(data, vq_vae_model, code_book_len)

    plot_comp_curves2(loss_history, savepath+"lr_ganvae.png")
    plot_comp_curves2(acc_h, savepath+"acc_ganvae.png")

    sample_vae_pt = GAN_vae.generate_image(100)

    gen_images = vq_vae.Decoder(sample_vae_pt)
    save_grid(gen_images.numpy(), save_gen_images)


#generate_layer_vae("/data/mainak/ADRL_E9_333/Data/tiny-imagenet-200/train.npy", "/data/mainak/ADRL_E9_333/VQ_VAE_32/model/", 32, "plot_save_latentvaelrcurve_latest.png", "layer_vq32_vae_gen_image_latest.jpg")
#DriverCode("/data/mainak/ADRL_E9_333/Data/tiny-imagenet-200/train.npy", "/data/mainak/ADRL_E9_333/Data/tiny-imagenet-200/cv.npy", "/data/mainak/ADRL_E9_333/VQ_VAE_512_new/model/", "/data/mainak/ADRL_E9_333/VQ_VAE_512_new/", 64)


#DriverCode("/data/mainak/ADRL_E9_333/Data/tiny-imagenet-200/train.npy", "/data/mainak/ADRL_E9_333/Data/tiny-imagenet-200/cv.npy", "/data/mainak/ADRL_E9_333/VQ_VAE_32/model/", "/data/mainak/ADRL_E9_333/VQ_VAE_32/", 32)

#generate_layer_GAN("/data/mainak/ADRL_E9_333/Data/tiny-imagenet-200/train.npy", "/data/mainak/ADRL_E9_333/VQ_VAE_32/model/", 32, "./", "layer_vq32_vae_gen_image_latest.jpg")
