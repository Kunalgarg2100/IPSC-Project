
# coding: utf-8

# In[5]:


import mnist
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import argparse
import cupy as cp

# In[6]:


def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def conv_forward_pass(image, filt, bias):
    # print("CUDA cODE")
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions

    convfilter_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void my_conv(const float* img, const float * filt, const float * bias, float * out, const int depth, const int img_dim, const int f, const int out_dim ) {
            const unsigned int j = blockDim.x*blockIdx.x + threadIdx.x;
            const unsigned int i = blockDim.y*blockIdx.y + threadIdx.y;
            const unsigned int pp = blockIdx.z;
            const unsigned int dp = depth;
            //printf("%d,%d ::: ",i,j);

            if(i<img_dim && j<img_dim){
                unsigned int oPixelPos = i*out_dim + j+ pp*out_dim*out_dim;
                out[oPixelPos] = 0;
                for(int h=0;h<dp;h++){
                    for(int k=0;k<f;k++){
                        for(int l=0;l<f;l++){
                            unsigned int iPixelPos = ((i+k)*img_dim + (j+l)) + img_dim*img_dim*h;
                            unsigned int filtPos = (k*f+l) + f*f*h + f*f*dp*pp;
                            out[oPixelPos] += img[iPixelPos] * filt[filtPos] + bias[pp];
                        }
                    }
                }
            }
        }
        ''', 'my_conv')

    # print(filt.shape)
    # for curr_f in range(n_f):
        # filt[curr_f]
    out_dim = int(in_dim - f)+1 # calculate output dimensions
    img_gpu = cp.asarray(image, dtype=cp.float32) #28X28
    img_dim_gpu = cp.int32(in_dim) #28
    filt_size_gpu = cp.int32(f) # 5
    out_dim_gpu = cp.int32(out_dim) #24
    depth_gpu = cp.int32(n_c)
    threads_per_block = f
    num_blocks = (in_dim//f) +1

    filt_gpu = cp.asarray(filt, dtype=cp.float32) #5X5
    # print(filt)
    # filt_gpu = cp.zeros((n_f, n_c_f, f, f), dtype=cp.float32) #5X5
    filt_gpu = cp.asarray(filt_gpu.flatten(), dtype=cp.float32)
    # print(filt_gpu)


    # out_gpu = cp.zeros((out_dim,out_dim,n_f), dtype=cp.float32) # 24 X 24
    out_gpu = cp.zeros((n_f,out_dim,out_dim), dtype=cp.float32) # 24 X 24
    out_gpu = out_gpu.flatten()

    bias_gpu = cp.asarray(bias, dtype=cp.float32) #5X5
    bias_gpu = bias_gpu.flatten()

    # print(filt_gpu.)

    # threads_per_block = 1
    # num_blocks = 1
   
    # print(dp)










    # print(depth_gpu, img_dim_gpu, filt_size_gpu, out_dim_gpu)


    # for curr_f in range(n_f):
        # curr_f = 7
    convfilter_kernel((num_blocks,num_blocks,8), (threads_per_block,threads_per_block), (img_gpu, filt_gpu, bias_gpu, out_gpu, depth_gpu, img_dim_gpu, filt_size_gpu, out_dim_gpu))
    # # convfilter_kernel((576,), (24,5), (img_gpu, filt_gpu, out_gpu, depth_gpu, img_dim_gpu, filt_size_gpu, out_dim_gpu))
    out_gpu = cp.reshape(out_gpu,(n_f,out_dim,out_dim))
    output = cp.asnumpy(out_gpu)
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
num_images = 10
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

