import numpy as np 
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

num_p = 50
num_p2 = 10
num_time = int(num_p/num_p2)
num_it = 5
delta = 80
n_G = 2
csv = np.array(pd.read_csv('cube_defect.csv'))

def detect(matrix, num, val):
  cols1 = matrix.shape[0]
  matrix_new = np.ones([cols1])
  for i in range(cols1):
      if matrix[i] == 0:
          if val == 1:
              matrix_new[i-(int(num/2)+1):i+int(num/2)] = 0
          else:
              matrix_new[i-(int(num/2)+1):i+int(num/2)] = 0
  return matrix_new

def detect_new(binary_matrix, num, cols1, cols2):
  cols0 = binary_matrix.shape[0]
  matrix_new = np.ones([cols0, cols1, cols2])
  for i in range(cols0):
      if binary_matrix[i] == 0:
          unif_random_matrix = np.random.uniform(0., 1., size = [int(cols1-num)])          
          binary_matrix2 = 1*(unif_random_matrix < np.max(unif_random_matrix))
          if num%2 == 0:
              binary_new = np.hstack((np.ones([int(num/2)]), binary_matrix2, np.ones([int(num/2)])))
              binary_final = np.tile(np.expand_dims(detect(binary_new, num,1), axis = 1), cols2)
          else:
              binary_new = np.hstack((np.ones([int(num/2)+1]), binary_matrix2, np.ones([int(num/2)])))
              binary_final = np.tile(np.expand_dims(detect(binary_new, num,0), axis = 1), cols2)              
          matrix_new[i,:,:] = binary_final + 0
  return matrix_new

def binary_sampler_window(num, data):
  cols0, cols1, cols2 = data.shape
  binary_random_matrix = np.zeros([cols0])    
  binary_m = detect_new(binary_random_matrix, num, cols1, cols2)
  return binary_m

def normalization_new(data):
  # Parameters
  _, dim1 = data.shape
  norm_data = data.copy()
  
  # MixMax normalization
  min_val = np.zeros([dim1])
  max_val = np.zeros([dim1])
  
  # For each dimension
  for i in range(dim1):
    min_val[i] = np.nanmin(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
    max_val[i] = np.nanmax(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
    
  # Return norm_parameters for renormalization
  norm_parameters = {'min_val': min_val,
                     'max_val': max_val}
      
  return norm_data, norm_parameters

csv_new, _ = normalization_new(csv) 

cols1 = csv_new.shape[0]
X_pos = csv_new[int(cols1*0.05):int(cols1*0.35),:]
X_neg = csv_new[int(cols1*0.6):int(cols1*0.9),:]

y_pos = np.ones([X_pos.shape[0]])
y_neg = np.zeros([X_neg.shape[0]])

X_pos1 = X_pos.shape[0]
X_neg1 = X_neg.shape[0]

X_train = np.vstack((X_pos[0:int(0.8*X_pos1),:],X_neg[0:int(0.8*X_neg1),:]))
y_train = np.hstack((y_pos[0:int(0.8*X_pos1)],y_neg[0:int(0.8*X_neg1)]))
X_test = np.vstack((X_pos[int(0.8*X_pos1):X_pos1,:],X_neg[int(0.8*X_neg1):X_neg1,:]))
y_test = np.hstack((y_pos[int(0.8*X_pos1):X_pos1],y_neg[int(0.8*X_neg1):X_neg1]))
data_impute = X_test + 0.0
    
fscore_new = np.zeros([7,100])
accuracy_new = np.zeros([7,100])
miss_val_list = [10, 15, 20, 25, 30, 35, 40]
it_new = [1000]

n_rs = 0
n_new = 0

for num in miss_val_list[:]:

    sub_csv = data_impute[0:num_p,:]
    sub_csv = np.expand_dims(sub_csv, axis=0)
    new_csv = sub_csv
    num_window = int(data_impute.shape[0]/num_p2)-num_time+1

    for i in range(1, num_window):
        sub_csv = data_impute[i*num_p2:(i+num_time)*num_p2,:]
        sub_csv = np.expand_dims(sub_csv, axis=0)
        new_csv = np.concatenate((new_csv, sub_csv), axis = 0)
        
    data_m_n = binary_sampler_window(num, new_csv)        
    
    data_new = new_csv + 0.0
    data_m = data_m_n + 0
    
    norm_data = data_new + 0.0
    
    img_i = 0
    
    df_ori = norm_data + 0.0
    
    no, dim1, dim2 = df_ori.shape

    imputed_data_all_new = np.zeros([no*num_p, data_impute.shape[1]])    
    
    dk = np.float32(dim1*dim2)
    
    dim = dim2 + 0
    
    batch_num = int(no/2)
        
    def binary_sampler(p, rows, cols1, cols2):
      unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols1, cols2])
      binary_random_matrix2 = 1*(unif_random_matrix < p)
      return binary_random_matrix2
    
    def binary_sampler_new(p, cols1, cols2):
      unif_random_matrix = np.random.uniform(0., 1., size = [cols1, cols2])
      binary_random_matrix2 = 1*(unif_random_matrix < p)
      return binary_random_matrix2
        
    def mae_loss (ori_data, imputed_data, data_m):
      nominator = np.sum(abs((1-data_m) * ori_data - (1-data_m) * imputed_data))
      denominator = np.sum(1-data_m)
      mae = nominator/float(denominator)
      return mae    
    
    def rmse_loss(ori_data, imputed_data, data_m):
      nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
      denominator = np.sum(1-data_m)
      rmse = np.sqrt(nominator/float(denominator))
      return rmse
    
    def renormalization(norm_data, norm_parameters):
      min_val = norm_parameters['min_val']
      max_val = norm_parameters['max_val']
    
      _, dim1, dim2 = norm_data.shape
      renorm_data = norm_data.copy()
        
      for i in range(dim1):
          for j in range(dim2):
              renorm_data[:,i,j] = renorm_data[:,i,j] * (max_val[i,j] + 1e-6)   
              renorm_data[:,i,j] = renorm_data[:,i,j] + min_val[i,j]
        
      return renorm_data
    
    def uniform_sampler(low, high, cols1, cols2):
      return np.random.uniform(low, high, size = [cols1, cols2])   
                      
    miss_data_x = np.array(df_ori)
    miss_data_x[data_m == 0] = np.nan            

    data_x = miss_data_x + 0.0

    #defining features and outcomes to train GAIN
    rs = 11
    
    batch_size = 1
    alpha = 100
    iterations = it_new[0]
  
    def xavier_init(size):
      in_dim = size[0]
      xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
      return tf.random_normal(shape = size, stddev = xavier_stddev)
    
    def sample_batch_index(total, batch_size):
      total_idx = np.random.permutation(total)
      batch_idx = total_idx[:batch_size]
      return batch_idx

    # Hidden state dimensions
    h_dim = int(dim)
    
    norm_data_x = np.nan_to_num(data_x, 0)        
    ## GAIN 1 architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [dim1, dim2])
    X2 = tf.placeholder(tf.float32, shape = [dim1, dim2])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape = [dim1, dim2])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [dim1, dim2])
      
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
      
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
      
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
      
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
      
    #Generator variables 0
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W2 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W3 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W4 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W5 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W6 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W7 = tf.Variable(xavier_init([dim*4, dim*2]))        
    G_W8 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_b8 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_b9 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_W9 = tf.Variable(xavier_init([dim*2, dim*2]))

    G_W10 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W11 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W12 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W13 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W14 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W15 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W16 = tf.Variable(xavier_init([dim*4, dim*2])) 
    G_W17 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W18 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W19 = tf.Variable(xavier_init([dim*4, dim*2]))
    G_W20 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_b20 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_b21 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_W21 = tf.Variable(xavier_init([dim*2, dim*2]))

    G_W22 = tf.Variable(xavier_init([dim*2, dim]))
    G_b22 = tf.Variable(tf.zeros(shape = [dim]))
    
    theta_G0 = [G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_W7, G_W8, G_W9, G_b8, G_b9,
                G_W10, G_W11, G_W12, G_W13, G_W14, G_W15, G_W16, G_W17, G_W18, 
                G_W19, G_W20, G_b20, G_W21,G_b21, G_W22, G_b22]

    #Generator variables 1
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W2_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W3_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W4_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W5_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W6_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W7_1 = tf.Variable(xavier_init([dim*4, dim*2]))        
    G_W8_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_b8_1 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_b9_1 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_W9_1 = tf.Variable(xavier_init([dim*2, dim*2]))

    G_W10_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W11_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W12_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W13_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W14_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W15_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W16_1 = tf.Variable(xavier_init([dim*4, dim*2])) 
    G_W17_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W18_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_W19_1 = tf.Variable(xavier_init([dim*4, dim*2]))
    G_W20_1 = tf.Variable(xavier_init([dim*2, dim*2]))
    G_b20_1 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_b21_1 = tf.Variable(tf.zeros(shape = [dim*2]))
    G_W21_1 = tf.Variable(xavier_init([dim*2, dim*2]))

    G_W22_1 = tf.Variable(xavier_init([dim*2, dim]))
    G_b22_1 = tf.Variable(tf.zeros(shape = [dim]))
    
    theta_G1 = [G_W1_1, G_W2_1, G_W3_1, G_W4_1, G_W5_1, G_W6_1, G_W7_1, G_W8_1, G_W9_1, G_b8_1, 
                G_b9_1, G_W10_1, G_W11_1, G_W12_1, G_W13_1, G_W14_1, G_W15_1, G_W16_1, G_W17_1, 
                G_W18_1, G_W19_1, G_W20_1, G_b20_1, G_W21_1, G_b21_1, G_W22_1, G_b22_1]

    def filter_layer(x, y, m, delta=80):
        new_dist = tf.sqrt(tf.reduce_sum((x*m)*(x*m)-2*(x*m)*(y*m)+(y*m)*(y*m),axis=1))
        dist_num = tf.contrib.distributions.percentile(new_dist, delta)
        w = tf.gather(x,tf.where(new_dist <= dist_num)[:,0])
        w_order = tf.where(new_dist <= dist_num)[:,0]
        return w, w_order
      
    ## GAIN functions
    # Generator
    def generator(x,x2,m,h):
      # Concatenate Mask and Data
      inputs = tf.concat(values = [x, m], axis = 1)
      Q1 = tf.matmul(inputs, G_W1)          
      Q2 = tf.matmul(inputs, G_W4)
      K1 = tf.matmul(inputs, G_W2)
      K2 = tf.matmul(inputs, G_W5)
      V1 = tf.matmul(inputs, G_W3)
      V2 = tf.matmul(inputs, G_W6)
      G_h1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q1,tf.transpose(K1)),1/tf.sqrt(dk))),V1)
      G_h2 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q2,tf.transpose(K2)),1/tf.sqrt(dk))),V2)
      G_h3 = tf.concat(values = [G_h1, G_h2], axis = 1)
      G_h4 = tf.add(tf.matmul(G_h3, G_W7),inputs)
      mean, var = tf.nn.moments(G_h4, axes=[0])
      G_h5 = tf.nn.batch_normalization(G_h4,mean,var,None,None,variance_epsilon=1e-6)
      G_h6 = tf.matmul(tf.nn.relu(tf.matmul(G_h5, G_W8) + G_b8), G_W9) + G_b9
      G_h7 = tf.add(G_h5,G_h6)
      mean, var = tf.nn.moments(G_h7, axes=[0])
      G_h8 = tf.nn.batch_normalization(G_h7,mean,var,None,None,variance_epsilon=1e-6)
      
      Q3 = tf.matmul(inputs, G_W10)   
      Q4 = tf.matmul(inputs, G_W11)   
      K3 = tf.matmul(inputs, G_W12)   
      K4 = tf.matmul(inputs, G_W13)   
      V3 = tf.matmul(inputs, G_W14)   
      V4 = tf.matmul(inputs, G_W15)   
      G_h9 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q3,tf.transpose(K3)),1/tf.sqrt(dk))),V3)
      G_h10 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q4,tf.transpose(K4)),1/tf.sqrt(dk))),V4)
      G_h11 = tf.concat(values = [G_h9, G_h10], axis = 1)
      G_h12 = tf.add(tf.matmul(G_h11, G_W16),inputs)
      mean, var = tf.nn.moments(G_h12, axes=[0])
      G_h13 = tf.nn.batch_normalization(G_h12,mean,var,None,None,variance_epsilon=1e-6)
      
      Q5 = tf.matmul(G_h13, G_W17)   
      Q6 = tf.matmul(G_h13, G_W18)  
      K5 = tf.matmul(G_h8, G_W2)   
      K6 = tf.matmul(G_h8, G_W5)   
      V5 = tf.matmul(G_h8, G_W3)   
      V6 = tf.matmul(G_h8, G_W6)           
      G_h14 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q5,tf.transpose(K5)),1/tf.sqrt(dk))),V5)
      G_h15 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q6,tf.transpose(K6)),1/tf.sqrt(dk))),V6)
      G_h16 = tf.concat(values = [G_h14, G_h15], axis = 1)
      G_h17 = tf.add(tf.matmul(G_h16, G_W19),inputs)
      mean, var = tf.nn.moments(G_h17, axes=[0])
      G_h18 = tf.nn.batch_normalization(G_h17,mean,var,None,None,variance_epsilon=1e-6)
      G_h19 = tf.matmul(tf.nn.relu(tf.matmul(G_h18, G_W20) + G_b20), G_W21) + G_b21
      G_h20 = tf.add(G_h19,G_h18)
      mean, var = tf.nn.moments(G_h20, axes=[0])
      G_h21 = tf.nn.batch_normalization(G_h20,mean,var,None,None,variance_epsilon=1e-6)       
      G_h22 = tf.nn.sigmoid(tf.matmul(G_h21, G_W22) + G_b22)

      # Concatenate Mask and Data
      inputs_1 = tf.concat(values = [x2, m], axis = 1)
      Q1_1 = tf.matmul(inputs_1, G_W1_1)          
      Q2_1 = tf.matmul(inputs_1, G_W4_1)
      K1_1 = tf.matmul(inputs_1, G_W2_1)
      K2_1 = tf.matmul(inputs_1, G_W5_1)
      V1_1 = tf.matmul(inputs_1, G_W3_1)
      V2_1 = tf.matmul(inputs_1, G_W6_1)
      G_h1_1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q1_1,tf.transpose(K1_1)),1/tf.sqrt(dk))),V1_1)
      G_h2_1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q2_1,tf.transpose(K2_1)),1/tf.sqrt(dk))),V2_1)
      G_h3_1 = tf.concat(values = [G_h1_1, G_h2_1], axis = 1)
      G_h4_1 = tf.add(tf.matmul(G_h3_1, G_W7_1),inputs_1)
      mean_1, var_1 = tf.nn.moments(G_h4_1, axes=[0])
      G_h5_1 = tf.nn.batch_normalization(G_h4_1,mean_1,var_1,None,None,variance_epsilon=1e-6)
      G_h6_1 = tf.matmul(tf.nn.relu(tf.matmul(G_h5_1, G_W8_1) + G_b8_1), G_W9_1) + G_b9_1
      G_h7_1 = tf.add(G_h5_1,G_h6_1)
      mean_1, var_1 = tf.nn.moments(G_h7_1, axes=[0])
      G_h8_1 = tf.nn.batch_normalization(G_h7_1,mean_1,var_1,None,None,variance_epsilon=1e-6)
      
      Q3_1 = tf.matmul(inputs_1, G_W10_1)   
      Q4_1 = tf.matmul(inputs_1, G_W11_1)   
      K3_1 = tf.matmul(inputs_1, G_W12_1)   
      K4_1 = tf.matmul(inputs_1, G_W13_1)   
      V3_1 = tf.matmul(inputs_1, G_W14_1)   
      V4_1 = tf.matmul(inputs_1, G_W15_1)   
      G_h9_1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q3_1,tf.transpose(K3_1)),1/tf.sqrt(dk))),V3_1)
      G_h10_1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q4_1,tf.transpose(K4_1)),1/tf.sqrt(dk))),V4_1)
      G_h11_1 = tf.concat(values = [G_h9_1, G_h10_1], axis = 1)
      G_h12_1 = tf.add(tf.matmul(G_h11_1, G_W16_1),inputs_1)
      mean_1, var_1 = tf.nn.moments(G_h12_1, axes=[0])
      G_h13_1 = tf.nn.batch_normalization(G_h12_1,mean_1,var_1,None,None,variance_epsilon=1e-6)
      
      Q5_1 = tf.matmul(G_h13_1, G_W17_1)   
      Q6_1 = tf.matmul(G_h13_1, G_W18_1)  
      K5_1 = tf.matmul(G_h8_1, G_W2_1)   
      K6_1 = tf.matmul(G_h8_1, G_W5_1)   
      V5_1 = tf.matmul(G_h8_1, G_W3_1)   
      V6_1 = tf.matmul(G_h8_1, G_W6_1)           
      G_h14_1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q5_1,tf.transpose(K5_1)),1/tf.sqrt(dk))),V5_1)
      G_h15_1 = tf.matmul(tf.nn.softmax(tf.multiply(tf.matmul(Q6_1,tf.transpose(K6_1)),1/tf.sqrt(dk))),V6_1)
      G_h16_1 = tf.concat(values = [G_h14_1, G_h15_1], axis = 1)
      G_h17_1 = tf.add(tf.matmul(G_h16_1, G_W19_1),inputs)
      mean_1, var_1 = tf.nn.moments(G_h17_1, axes=[0])
      G_h18_1 = tf.nn.batch_normalization(G_h17_1,mean_1,var_1,None,None,variance_epsilon=1e-6)
      G_h19_1 = tf.matmul(tf.nn.relu(tf.matmul(G_h18_1, G_W20_1) + G_b20_1), G_W21_1) + G_b21_1
      G_h20_1 = tf.add(G_h19_1,G_h18_1)
      mean_1, var_1 = tf.nn.moments(G_h20_1, axes=[0])
      G_h21_1 = tf.nn.batch_normalization(G_h20_1,mean_1,var_1,None,None,variance_epsilon=1e-6)       
      G_h22_1 = tf.nn.sigmoid(tf.matmul(G_h21_1, G_W22_1) + G_b22_1)
      
      return G_h22, G_h22_1
          
    # Discriminator
    def discriminator(x, h):
      # Concatenate Data and Hint
      inputs = tf.concat(values = [x, h], axis = 1) 
      D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
      D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
      D_logit = tf.matmul(D_h2, D_W3) + D_b3
      D_prob = tf.nn.sigmoid(D_logit)
      return D_prob
    
    G_sample, G_sample2 = generator(X, X2, M, H)
    
    Sam_new, sam_order = filter_layer(G_sample, X, M, delta)
    Sam_new2, sam_order2 = filter_layer(G_sample2, X, M, delta)
    
    # Combine with observed data
    X_new = tf.gather(X,sam_order)
    M_new = tf.gather(M,sam_order)
    Hat_X = X_new * M_new + Sam_new * (1-M_new)
    H_new = tf.gather(H,sam_order)        

    X_new2 = tf.gather(X,sam_order2)
    M_new2 = tf.gather(M,sam_order2)
    Hat_X2 = X_new2 * M_new2 + Sam_new2 * (1-M_new2)
    H_new2 = tf.gather(H,sam_order2)  
    
    # Discriminator
    D_prob1 = discriminator(Hat_X, H_new)
    D_prob2 = discriminator(Hat_X2, H_new2)
      
    ## GAIN loss
    D_loss_temp = ((-tf.reduce_mean(M_new * tf.log(D_prob1 + 1e-8) + (1-M_new) * tf.log(1. - D_prob1 + 1e-8)))+
                    (-tf.reduce_mean(M_new2 * tf.log(D_prob2 + 1e-8) + (1-M_new2) * tf.log(1. - D_prob2 + 1e-8))))/n_G
     
    G_loss_temp1 = -tf.reduce_mean((1-M_new) * tf.log(D_prob1 + 1e-8))
    G_loss_temp2 = -tf.reduce_mean((1-M_new2) * tf.log(D_prob2 + 1e-8))
      
    MSE_loss1 = tf.reduce_mean((M_new * X_new - M_new * Sam_new)**2) / tf.reduce_mean(M_new)
    MSE_loss2 = tf.reduce_mean((M_new2 * X_new2 - M_new2 * Sam_new2)**2) / tf.reduce_mean(M_new2)
      
    D_loss = D_loss_temp
    G_loss1 = G_loss_temp1 + alpha * MSE_loss1 
    G_loss2 = G_loss_temp2 + alpha * MSE_loss2
      
    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver1 = tf.train.AdamOptimizer().minimize(G_loss1, var_list=theta_G0)
    G_solver2 = tf.train.AdamOptimizer().minimize(G_loss2, var_list=theta_G1)
    
    for num_j in range(num_it):
        hint_rate = 0.55+0.1*num_j
        ## Iterations
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
           
        D = np.array(range(0,iterations), dtype = "float64")
        G1 = np.array(range(0,iterations), dtype = "float64")
        G2 = np.array(range(0,iterations), dtype = "float64")
        
        # Start Iterations
        for ite in tqdm(range(iterations)):    
              
            # Sample batch
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = np.squeeze(data_x[batch_idx, :, :])
            M_mb = np.squeeze(data_m[batch_idx, :, :])  
            # Sample random vectors  
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim2) 
            Z_mb2 = uniform_sampler(0, 0.01, batch_size, dim2)
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim1, dim2)
            H_mb = np.squeeze(M_mb * H_mb_temp)
              
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            X_mb2 = M_mb * X_mb + (1-M_mb) * Z_mb2  

            _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                      feed_dict = {M: M_mb, X: X_mb, X2: X_mb2, H: H_mb})
                  
            _, G_loss_curr1, MSE_loss_curr1 = sess.run([G_solver1, G_loss1, MSE_loss1],
                     feed_dict = {M: M_mb, X: X_mb, X2: X_mb2, H: H_mb})

            _, G_loss_curr2, MSE_loss_curr2 = sess.run([G_solver2, G_loss_temp2, MSE_loss2],
                     feed_dict = {M: M_mb, X: X_mb, X2: X_mb2, H: H_mb})
            
            D[ite] = D_loss_curr
            G1[ite] = G_loss_curr1
            G2[ite] = G_loss_curr2
    
        imputed_data_all = np.zeros([no*num_p, data_impute.shape[1]])
        y_test_new = np.zeros([no*num_p])
        X_test_new = np.zeros([no*num_p, data_impute.shape[1]])

        for num_i in range(0, no):        
            Z_mb = uniform_sampler(0,1,dim1,dim2) 
            Z_mb2 = uniform_sampler(0,1,dim1,dim2) 
            M_mb = np.squeeze(data_m[num_i,:,:])
            X_mb = np.squeeze(data_x[num_i,:,:])          
            X_mb2 = np.squeeze(data_x[num_i,:,:])          
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
            X_mb2 = M_mb * X_mb + (1-M_mb) * Z_mb2
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim1, dim2)
            H_mb = np.squeeze(M_mb * H_mb_temp)
            
            imputed_data_new1, imputed_data_new2 = sess.run([G_sample, G_sample2], 
                                                            feed_dict = {X: X_mb, X2: X_mb2, M: M_mb, H: H_mb})
            
            imputed_data_new = (imputed_data_new1 + imputed_data_new2)/2
              
            imputed_data = M_mb * X_mb + (1-M_mb) * imputed_data_new
            imputed_data_all[int(num_i*num_p):int(num_i*num_p+num_p),:] = imputed_data + 0.0
            y_test_new[int(num_i*num_p):int(num_i*num_p+num_p)] = y_test[int(num_i*num_p2):int(num_i*num_p2+num_p)]
            X_test_new[int(num_i*num_p):int(num_i*num_p+num_p),:] = X_test[int(num_i*num_p2):int(num_i*num_p2+num_p),:]
            
        imputed_data_all_new = imputed_data_all_new + imputed_data_all
        
    imputed_data_all = imputed_data_all_new/num_it

    n_new = n_new + 1
    clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    for num_i in range(0, no):            
        pre_y_test = clf.predict(imputed_data_all[int(num_i*num_p):int(num_i*num_p+num_p),:])
        calculate_report = precision_recall_fscore_support(y_test_new[int(num_i*num_p):int(num_i*num_p+num_p)], pre_y_test, average = 'weighted')
        fscore = round(calculate_report[2], 3) 
        accuracy = accuracy_score(y_test_new[int(num_i*num_p):int(num_i*num_p+num_p)], pre_y_test)
        fscore_new[n_new-1,num_i] = fscore
        accuracy_new[n_new-1,num_i] = accuracy           
            
print(np.mean(fscore_new[:,0:no], axis = 1))
 
