import numpy as np
import matplotlib.pyplot as plt
#from Diffusion_Model_Q1 import *
from Diffusion_Model_final import *
import tensorflow as tf
import pandas as pd


def get_10_attributes(attr_file, index_list):
    attrs = pd.read_csv(attr_file)
    att_mat = attrs[index_list].to_numpy()
    column_names = list(attrs[index_list].columns)
    return att_mat, column_names
    

def write_attributes(loc, writeloc):
    df = pd.read_table(loc, sep="\t")
    feat = []
    
    col_names = df.iloc[0][0].split(" ")[0:40]
    print(col_names)
    for i in range(1, len(df)):
        s = s = df.iloc[i][0].split(" ")[1: ]
        feat.append((list(int((int(c)+1)/2) for c in s if c in ['-1', '1'])))
    
    feat_map = np.array(feat)
    print(feat_map.shape)
    feat_map = pd.DataFrame(feat_map, columns=col_names)
    
    feat_map.to_csv(writeloc, index=False)

#write_attributes("../CelebA/Anno/list_attr_celeba.txt", "../CelebA/attributes.csv")

def preprocess_image(data, sz):
    all_imgs = [] 
    for i in range(data.shape[0]):
        img = tf.image.resize(data[i], size=[sz, sz], antialias=True)
        img = tf.clip_by_value(img / 255.0, 0.0, 1.0).numpy()
        #print(img.shape)
        all_imgs.append(img)
    
    all_imgs = np.array(all_imgs)
    return all_imgs
    
def plot_comp_curves(losses, __title__, y_label, index, label, saveloc):
    plt.figure()
    plt.plot(losses[index[0]], label=label[0])
    plt.plot(losses[index[1]], label=label[1])
    
    plt.legend()
    plt.xlabel('Epoch', size=15)
    plt.ylabel(y_label, size=15)
    plt.title(__title__, size=15)
    plt.savefig(saveloc)
    
    #plt.show()

def plot_comp_curves_1(loss, __title__, y_label, saveloc):
    plt.figure()
    plt.plot(loss)
    
    plt.xlabel('Epoch', size=15)
    plt.ylabel(y_label, size=15)
    plt.title(__title__, size=15)
    plt.savefig(saveloc)
    
    #plt.show()
    
def save_grid(data, savePath, grid_specs, incr):
    from PIL import Image
    #tiny image is 64x64 
    grid_10_10 = Image.new('RGB', (grid_specs[1], grid_specs[0]))
    idx_down = 0
    idx_side = 0
    for i in range(0, grid_specs[1], int(grid_specs[1]/incr[1])):
        idx_down = 0
        for j in range(0, grid_specs[0], int(grid_specs[0]/incr[0])):
            im = data[idx_side][idx_down]
            im = Image.fromarray(im)
            grid_10_10.paste(im, (i,j))
            idx_down += 1
        
        idx_side +=1
    grid_10_10.save(savePath)


def save_grid2(data, savePath, grid_specs, incr, col_names):
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
    #tiny image is 64x64 
    offset = 100
    grid_10_10 = Image.new('RGB', (grid_specs[1] + offset, grid_specs[0]+offset))
    
    ctr = 0
    for i in range(offset, grid_specs[0]+offset, int(grid_specs[0]/incr[0])):
        for j in range(offset, grid_specs[1]+offset, int(grid_specs[1]/incr[1])):
            im = data[ctr]
            im = Image.fromarray(im)
            #coordinate system is (x: rightways, y: downways)
            grid_10_10.paste(im, (j,i))
            ctr += 1
        
    
    I_write = ImageDraw.Draw(grid_10_10)
    
    # Custom font style and font size
    myFont1 = ImageFont.truetype('Arial.ttf', 15)
    myFont2 = ImageFont.truetype('Arial.ttf', 35)
    #display the labels
    ctr = 1
    for i in range(int(offset + grid_specs[0]/(2*incr[1])), int(grid_specs[1] + offset), int(grid_specs[1]/incr[1])):
        #display the classes, write with black
        I_write.text((0, i), col_names[ctr], font=myFont1, fill=(255,255,255))
        ctr += 1
        
    I_write.text((int(0.22*grid_specs[1] + offset), int(0.4*offset)), "Female", font=myFont2, fill=(255,255,255))
    I_write.text((int(0.70*grid_specs[1] + offset), int(0.4*offset)), "Male", font=myFont2, fill=(255,255,255))
    
    ends = (int(offset + grid_specs[0]/2), 0), (int(offset + grid_specs[0]/2), grid_specs[1]+100)
    I_write.line(ends, fill=(255,255,255), width=8)
    
    grid_10_10.save(savePath)

#this is used to create the 2 dataset
def generate_images(dataloc, model_loc, r_save, f_save, T):
    from PIL import Image
    n_loop = 10
    d_size = 100
    
    
    data = np.load(dataloc)
    data = (data[np.random.choice(data.shape[0], 1000, replace=False)]).astype('float32')
    data = preprocess_image(data[0:1000], 128)
    
    my_diff_model_noise_form = Diffusion_Model_modified_schedule(data.shape[1:], T)
    my_diff_model_noise_form.load_mymodel(model_loc)
    for i in range(n_loop*d_size):
        
        img = Image.fromarray((data[i]*255).astype('uint8'))
        img.save(r_save+str(i)+".png")
    
    for j in range(n_loop):
        langevian_steps = my_diff_model_noise_form.sample_images2(d_size)    
        for i in range(d_size):
            
            img = Image.fromarray(langevian_steps[i])
            img.save(f_save+str(j*1000 + i+1)+".png")

def train_diff_model_noise(dataloc, saveloc, T, n_epochs, train_model, inbuilt=False):
    data = np.load(dataloc)
    data = (data[np.random.choice(data.shape[0], 25000, replace=False)]).astype('float32')
    data /= 255.0
    print(data.shape)
    if train_model == True:
        my_diff_model_noise_form = Diffusion_Model_modified_schedule(data.shape[1:], T, inbuilt)
        my_diff_model_noise_form.make_model()
        #train the model
        loss_history = my_diff_model_noise_form.train_diff_model_noise_formulation(data, n_epochs)
    
    
        #plot_comp_curves(loss_history, "Learning Curves", "Loss", ["recons", "diff"], ["reconstruction", "noise matching"], saveloc+"/learning_curve.png")
        plot_comp_curves_1(loss_history, "Learning Curves", "Diffusion Loss",  saveloc+"learning_curve.png")
        #save and end training
        my_diff_model_noise_form.save_model(saveloc)
        #del(my_diff_model_noise_form)
    #load and start inference
    my_diff_model_noise_form = Diffusion_Model_modified_schedule(data.shape[1:], T)
    my_diff_model_noise_form.load_mymodel(saveloc)
    
    added_noise = (data[np.random.choice(25000, 10, replace=False)]).astype('float32')
    added_noise =  added_noise
    langevian_steps = my_diff_model_noise_form.sample_images(10, added_noise)
    langevian_steps2 = my_diff_model_noise_form.sample_images(10)
    
    #print(len(langevian_steps), langevian_steps[0].shape)
    save_grid(langevian_steps, saveloc+"lang_res_noise.jpg", [data.shape[1]*10, data.shape[2]*11], [10, 11])
    save_grid(langevian_steps2, saveloc+"lang_res_imageback.jpg", [data.shape[1]*10, data.shape[2]*11], [10, 11])
    

def train_diff_model_noise_celeba(dataloc, saveloc, T, n_epochs, train_model=True, inbuilt=False):
    data = np.load(dataloc)
    data = (data[np.random.choice(data.shape[0], 30000, replace=False)]).astype('float32')
    data = preprocess_image(data, 128)
    #print(data.shape)
    if train_model==True:
        my_diff_model_noise_form = Diffusion_Model_modified_schedule(data.shape[1:], T, inbuilt)
        my_diff_model_noise_form.make_model()
        #train the model
        loss_history = my_diff_model_noise_form.train_diff_model_noise_formulation(data, n_epochs)


        #plot_comp_curves(loss_history, "Learning Curves", "Loss", ["recons", "diff"], ["reconstruction", "noise matching"], saveloc+"/learning_curve.png")
        plot_comp_curves_1(loss_history, "Learning Curves", "Diffusion Loss",  saveloc+"learning_curve.png")
        #save and end training
        my_diff_model_noise_form.save_model(saveloc)
        del(my_diff_model_noise_form)
    
    
    #load and start inference
    my_diff_model_noise_form = Diffusion_Model_modified_schedule(data.shape[1:], T)
    my_diff_model_noise_form.load_mymodel(saveloc)
    
    added_noise = data[np.random.choice(30000, 10, replace=False)]
    langevian_steps = my_diff_model_noise_form.sample_images(10, added_noise)
    langevian_steps2 = my_diff_model_noise_form.sample_images(10)
    
    #print(len(langevian_steps), langevian_steps[0].shape)
    save_grid(langevian_steps, saveloc+"lang_res_noise2.jpg", [data.shape[1]*10, data.shape[2]*11], [10, 11])
    save_grid(langevian_steps2, saveloc+"lang_res_imageback2.jpg", [data.shape[1]*10, data.shape[2]*11], [10, 11])

# number of examples per class per gender
def sample_ys_for_display(n, n_attris):
    #1 is added for not male and female being in a category (1st category)
    y = np.zeros((2*n*n_attris, n_attris+1))
    for j in range(n_attris):
        for i in range(2*n):
            if int(i/n) == 1:
                #male
                y[2*n*j+i, 0] = 1
            y[2*n*j+i, j+1] = 1
    return y


def train_diff_model_noise_celeba_classifier_guidance(dataloc, saveloc, T, n_epochs, train_model=True, inbuilt=False, n_attris=11):
    data = np.load(dataloc)
    choice_list = np.random.choice(data.shape[0], 30000, replace=False)
    data = data[choice_list].astype('float32')
    data = preprocess_image(data, 128)
    
    att_mat, col_names = get_10_attributes("../CelebA/attributes.csv", ["Male", "Bald", "Blond_Hair", "Black_Hair", "Chubby", 
                                                                        "Wearing_Hat", "Eyeglasses", "Smiling", "Wearing_Necklace",
                                                                        "Double_Chin", "Heavy_Makeup"])
    
    att_mat = att_mat[choice_list].astype('float32')
    
    #print(data.shape)
    if train_model==True:
        my_diff_model_noise_form = Diffusion_Model_classifier_guidance(data.shape[1:], T, n_attris, inbuilt)
        my_diff_model_noise_form.make_model()
        #train the model
        loss_history, acc_history = my_diff_model_noise_form.train_diff_model_noise_formulation(data, att_mat, n_epochs)


        plot_comp_curves(loss_history, "Learning Curves", "Loss", ["classifier", "diff"], ["classifier loss", "diffusion loss"], saveloc+"/learning_curve.png")
        plot_comp_curves_1(acc_history, "Accuracy Curves", "accuracy",  saveloc+"accuracy_curve.png")
        #save and end training
        my_diff_model_noise_form.save_model(saveloc)
        del my_diff_model_noise_form
    
        #number of examples per class
        y = sample_ys_for_display(1, n_attris-1)
        #choose 10 examples for langevin steps
        y = y[np.random.choice(2*(n_attris-1), 10, replace=False)]
        
        #load and start inference
        my_diff_model_noise_form = Diffusion_Model_classifier_guidance(data.shape[1:], T, n_attris)
        my_diff_model_noise_form.load_mymodel(saveloc)
        
        langevian_steps2 = my_diff_model_noise_form.sample_images(10, y)
        
        save_grid(langevian_steps2, saveloc+"lang_res_image.jpg", [data.shape[1]*10, data.shape[2]*11], [10, 11])
     
    else:
        y = sample_ys_for_display(5, n_attris-1)
        
        #load and start inference
        my_diff_model_noise_form = Diffusion_Model_classifier_guidance(data.shape[1:], T, n_attris)
        my_diff_model_noise_form.load_mymodel(saveloc)
        
        langevian_steps2 = my_diff_model_noise_form.sample_images2(100, y)
        save_grid2(langevian_steps2, saveloc+"class_conditioned_gen_n4.jpg", [data.shape[1]*10, data.shape[2]*10], [10, 10], col_names)

#train_diff_model_noise_celeba_classifier_guidance("../CelebA/celeba.npy", "./Diffusion_CelebA_1000_final_inbuiltunet_class/", 20, 10, train_model=True, inbuilt=True, n_attris=11)
#train_diff_model_noise_celeba_classifier_guidance("../CelebA/celeba.npy", "./Diffusion_CelebA_1000_final_inbuiltunet_class/", 20, 10, train_model=False, inbuilt=True, n_attris=11)   
#train_diff_model_noise("../../Assignment1/Data/bit-emojis/emojis.npy", "./Diff_model_bitemojis_n_20_final/", 20, 10, False, True)
#train_diff_model_noise_celeba("../CelebA/celeba.npy", "./Diffusion_CelebA_1000_final_inbuiltunet/", 1000, 10, True)
#generate_images("../../Assignment1/Data/bit-emojis/emojis.npy", "./Diff_model_bitemojis_n_20_final/", "./FID/bitemoji/Real/", "./FID/bitemoji/Fake/", 20)
#generate_images("../CelebA/celeba.npy", "./Diffusion_CelebA_20_final_inbuiltunet/", "./FID/celeba/Real/", "./FID/celeba/Fake/", 20)
#generate_images("../CelebA/celeba.npy", "./Diffusion_CelebA_20_final_inbuiltunet/", "./FID/ex/", "./FID/celeba/Fake/", 20)
#The FID score 
#!python -m pytorch_fid "./FID/Real/" "./FID/Fake" --device cuda:0 #: 129 (bitemoji), 280 (celeba)
