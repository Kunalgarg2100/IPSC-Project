
# coding: utf-8

# In[1]:


import mnist
import numpy as np
import cupy as cp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import argparse


# In[2]:


def modelParams():
    f1_shape = (8,1,5,5)
    f2_shape = (8,8,5,5)
    w3_shape = (128,800)
    w4_shape = (10,128)
    params = initializeParams(f1_shape,f2_shape,w3_shape,w4_shape)
    return params

def initializeParams(f1_shape,f2_shape,w3_shape,w4_shape):
    f1 = initializeFilter(f1_shape)
    f2 = initializeFilter(f2_shape)
    w3 = initializeWeight(w3_shape)
    w4 = initializeWeight(w4_shape)
    
    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((w3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))
    
    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    return params
    
def initializeFilter(size):
    stddev = 1.0/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))

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

def forward_pass(x,params):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    conv1 = conv_forward_pass(x, f1, b1)
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
    
    return conv1, conv2, fc1, fc2, out, probs


# In[8]:


def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def fc_backward_pass(dout, fc_layer, w, bshape):
    dw = dout.dot(fc_layer.T)
    db = np.sum(dout, axis = 1).reshape(bshape)
    df = w.T.dot(dout)
    return dw, db, df

def maxpool_backward_pass(dpool, orig):
    n_c, orig_dim, _ = orig.shape
    f , s = 2 ,2
    dout = np.zeros(orig.shape)
    
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout


def conv_backward_pass(dconv_prev, conv_in, filt, flag):
    num_filters, depth_f, fsize, fsize = filt.shape
    _, orig_dim, _ = conv_in.shape
    dp,out_dim,_ = dconv_prev.shape

    ## initialize derivatives
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((num_filters,1))
    tmpf = np.zeros(filt.shape)
    for curr_f in range(num_filters):
        for d in range(depth_f):
            tmpf[curr_f][d] = np.flip(filt[curr_f][d])

    padded_dconv = np.pad(dconv_prev,((0,0),(fsize-1,fsize-1),(fsize-1,fsize-1)))
    
    for curr_f in range(num_filters):
        for i in range(0,fsize):
            for j in range(0,fsize):
                dfilt[curr_f][:,i,j] = np.sum(dconv_prev[curr_f]*conv_in[:,i:i+out_dim,j:j+out_dim],axis=(1,2))
        dbias[curr_f] = np.sum(dconv_prev[curr_f])

    if flag:
        output = conv_forward_pass(padded_dconv, tmpf, np.zeros((num_filters,1)))
        return dfilt, dbias, output

    else:
        return dfilt, dbias

def backward_pass(x, label, params, conv1, conv2, fc1, fc2, out, probs):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    # Differentiation of loss with respect to out
    dout = probs - label
    dw4, db4, dfc2 = fc_backward_pass(dout, fc2, w4, b4.shape)
    dfc2[fc2<=0] = 0
    dw3, db3, dfc1 = fc_backward_pass(dfc2, fc1, w3, b3.shape)
    dfc1[fc1<=0] = 0
    
    out_dim = int(conv2.shape[1]/2)
    dpool = dfc1.reshape(conv2.shape[0],out_dim, out_dim)
    dconv2 = maxpool_backward_pass(dpool, conv2)
    dconv2[conv2<=0] = 0
    
    df2, db2, dconv1 = conv_backward_pass(dconv2, conv1, f2,1)
    dconv1[conv1<=0] = 0
    
    df1, db1 = conv_backward_pass(dconv1, x, f1,0)
    
    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]
    
    return grads


# In[9]:


def GD(batch_grads,params,batch_size):
    lr = 0.01
    beta1 , beta2 =  0.95, 0.99
#     '''
#     update the parameters through Adam gradient descnet.
#     '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    
    for i in range(batch_size):
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = batch_grads[i]
        
        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        dw3+=dw3_
        db3+=db3_
        dw4+=dw4_
        db4+=db4_


    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
   
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    return params


# In[10]:


num_images = 60000
batch_size = 32
X = np.asarray(mnist.train_images()[:num_images],dtype=np.float32)
y = mnist.train_labels()[:num_images]
num_classes = 10
imgdim = X.shape[1]

# Random Shuffle the data
permutation = np.random.permutation(num_images)
X = X[permutation]
y = y[permutation]

# Normalizing the images
X = X - int(np.mean(X))
X = X / int(np.std(X))
Y = np.array([np.eye(num_classes)[int(y[k])].reshape(num_classes, 1) for k in range(0,num_images)])
X = X.reshape(num_images, 1, imgdim, imgdim)

# Forming Batches for training
batches = [X[k:k+batch_size] for k in range(0,num_images,batch_size)]
batches_labels = [Y[k:k+batch_size] for k in range(0,num_images,batch_size)]
num_batches = len(batches)

params = pickle.load(open("iniParams.pkl", 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = params


# In[11]:


parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')
args = parser.parse_args()
save_path = args.save_path


# In[13]:


t = tqdm(batches)
loss_array = []
f = open(save_path, 'wb')

for i,batch in enumerate(t):        
    batch_labels = batches_labels[i]
    batch_grads = []
    cost = 0
    batch_size = len(batch_labels)
    for i in range(batch_size):
        img = batch[i]
        label = batch_labels[i]
        conv1, conv2, fc1, fc2, out, probs = forward_pass(img, params)
        grads = backward_pass(img, label, params, conv1, conv2, fc1, fc2, out, probs)
        batch_grads.append(grads)
        loss = categoricalCrossEntropy(probs, label)
        cost += loss
    cost = cost / batch_size
    params = GD(batch_grads,params,batch_size)
    # print(cost)`
    t.set_description("Cost: %.2f" % cost)
    loss_array.append(cost)
to_save = [params, loss_array]
pickle.dump(to_save, f)
f.close()


