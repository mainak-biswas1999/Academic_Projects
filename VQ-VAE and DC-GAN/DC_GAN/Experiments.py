from DC_GAN import *
import matplotlib.pyplot as plt

def save_grid(data, savePath):
    from PIL import Image
    #tiny image is 64x64 
    grid_10_10 = Image.new('RGB', (1280,1280))
    idx = 0
    for i in range(0, 1280, 128):
        for j in range(0, 1280, 128):
            im = data[idx]
            im = Image.fromarray(im)
            grid_10_10.paste(im, (i,j))
            idx += 1
            
    grid_10_10.save(savePath)


def plot_comp_curves(losses, __title__, y_label, label, saveloc, gen_cycle, type_div):
    plt.plot(losses[label[0]], label=label[0])
    plt.plot(losses[label[1]], label=label[1])
    
    plt.legend()
    plt.xlabel('Epoch', size=15)
    plt.ylabel(y_label, size=15)
    plt.title(__title__, size=15)
    plt.savefig(saveloc+y_label+type_div+str(gen_cycle)+"SameOpti.png")
    plt.clf()
    #plt.show()


def train_GAN(dataloc, lr_saveloc, img_saveloc, model_saveloc, gen_cycle, JS_flag):
    my_dc_gan = DC_GAN()
    my_dc_gan.makeModel()
    data = np.load(dataloc)

    loss_history, acc_history = my_dc_gan.train_GAN(data, 10, gen_cycle, JS_flag)
    plot_comp_curves(loss_history, "Loss of generator and Discriminator", "Loss", ['gen_loss', 'disc_loss'], img_saveloc, gen_cycle, "impro")
    plot_comp_curves(acc_history, "Accuracy of the Discriminator", "Accuracy", ['r_acc', 'f_acc'], lr_saveloc, gen_cycle, "impro")
    img_100 = my_dc_gan.GAN_generate(100)
    save_grid(img_100, img_saveloc+"Gen_Images_impro_genCycle_sameOpti"+str(gen_cycle)+".jpg")

    my_dc_gan.save_model(model_saveloc)

train_GAN("/home/mainakbiswas/ADRL_E9_333/Data/bit-emojis/emojis.npy", "./", "./", "./DC_GAN/", 3, 0)


#this is used to create the 2 dataset
def create_Comparison(dataloc, r_save, model_loc, f_save):
    from PIL import Image
    #save 1000 real
    d_size = 10000
    #data_path = os.listdir(dataloc)
    #selected_idx = np.random.choice(len(data_path), size=d_size, replace=False)
    #ctr = 1
    #for idx in selected_idx:
    #    img = Image.open(dataloc+data_path[idx])
    #    img.save(r_save+str(ctr)+".png")
    #    ctr += 1
    #1000 generated images
    emoji_generator = DC_GAN()
    emoji_generator.load_mymodel(model_loc)
    for j in range(10):
        generated_emojis = emoji_generator.GAN_generate(1000)
        for i in range(1000):
            img = Image.fromarray(generated_emojis[i])
            img.save(f_save+str(j*1000 + i+1)+".png")
#create_Comparison("../Data/bit-emojis/images/", "./FID/Real/", "./DC_GAN_3/", "./FID/Fake/")
#!python -m pytorch_fid "./FID/Real/" "./FID/Fake" --device cuda:0