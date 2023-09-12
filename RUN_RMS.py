#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mnist_loader1
import network34
import pickle
training_data, validation_data , test_data = mnist_loader1.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net=network34.Network([784,50,10])
net.SGD( training_data, 30, 10, 1.0, test_data=test_data)


# In[ ]:




