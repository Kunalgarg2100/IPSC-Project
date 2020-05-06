
# coding: utf-8

# In[5]:


import mnist
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import argparse


# In[6]:


def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def conv_forward_pass(image, filt, bias):
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/1)+1 # calculate output dimensions
    
    featMat = []
    kernelMat = []

    for j in range(in_dim-f+1):
        for k in range(in_dim-f+1):
            rowmat = []
            for i in range(n_c):
                currmat = image[i,j:j+f,k:k+f]
                currmat = currmat.flatten()
                # print(len(currmat))
                rowmat.extend(currmat.tolist())
            # print(len(rowmat))
            featMat.append(rowmat)
    featMat = np.array(featMat)

    for j in range(n_c_f):
        for k in range(f):
            for l in range(f):
                rowmat = []
                for i in range(n_f):
                    rowmat.append(filt[i,j,k,l])
                kernelMat.append(rowmat)
    kernelMat = np.array(kernelMat)

    output = np.matmul(featMat, kernelMat).T
    output = output.reshape((8,out_dim,out_dim))

    for curr_f in range(n_f):
        output[curr_f] = output[curr_f] + bias[curr_f]

    return output
                   
def maxpool_forward_pass(image):
    n_c, h_prev, w_prev = image.shape
    f , s = 2 ,2
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1
    
    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled
    
def fc_forward_pass(x, w, b):
    return w.dot(x) + b


# In[7]:


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4):
    '''
    Make predictions with trained filters/weights. 
    '''
    conv1 = conv_forward_pass(image, f1, b1)
    conv1[conv1<=0] = 0
                   
    conv2 = conv_forward_pass(conv1, f2, b2)
    conv2[conv2<=0] = 0
        
    pooled = maxpool_forward_pass(conv2)
    n_c, dim, _ = pooled.shape
    fc1 = pooled.reshape((n_c * dim * dim, 1)) # first dense layer
    
    fc2 = fc_forward_pass(fc1, w3, b3)
    fc2[fc2<=0] = 0
    
    out = fc_forward_pass(fc2, w4, b4)
    probs = softmax(out)
    
    return np.argmax(probs), np.max(probs)


# In[8]:


parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')
args = parser.parse_args()
save_path = args.save_path
params, cost = pickle.load(open(save_path, 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = params


# In[ ]:
num_images = 10000
X = np.asarray(mnist.test_images()[:num_images],dtype=np.float32)
Y = mnist.test_labels()[:num_images]
num_classes = 10
imgdim = X.shape[1]

# Normalizing the images
X = X - int(np.mean(X))
X = X / int(np.std(X))
X = X.reshape(num_images, 1, imgdim, imgdim)

t = tqdm(range(num_images), leave=True)
corr = 0

for i in t:
    x = X[i]
    label = Y[i]
    pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
    if pred==label:
        corr+=1
    t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))

