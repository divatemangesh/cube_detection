
# coding: utf-8

# In[5]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#if you do not add following two lines   then error may come is few version with tf with backend
#for more discription check "https://github.com/fchollet/keras/issues/3945"
from keras import backend as K
K.set_image_dim_ordering('th')


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[6]:



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[7]:


batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
#model.fit(x_train, y_train, epochs=5, batch_size=32)


# In[8]:


history=model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
#model.save_weights('first_try.h5')  # always save your weights after training or during training


# In[9]:


from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# later...
'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''


# In[27]:


'''import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
ip="data/train/cube/IMG_0099.jpg"
img = cv2.imread(ip)
img_res = cv2.imread(ip)
img = cv2.resize(img,(150, 150))
img = np.reshape(img,[1,3,150, 150])
classes = model.predict_classes(img)
cv2.putText(img_res,str(classes),(10,500), cv2.FONT_HERSHEY_SIMPLEX, 10,(255,0,0),10)

plt.imshow(img_res)
print classes

'''



# In[ ]:


'''import numpy as np
import cv2
#im = ["cf.jpg","ch.jpg","nc.jpg"]
im = [1,2,3]
res = []
for i in im :
    img = cv2.imread(im[i] )
    img = cv2.resize(img,(150, 150))
    img = np.reshape(img,[1,3,150, 150])
    classes = model.predict_classes(img)
    res.append((classes,im[i]))'''


# In[61]:


print history.history.keys()


# In[63]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[65]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[12]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[13]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[46]:





# In[79]:




#help(os.listdir)
#os.walk("data/train/cube/",onerror=None)
list_train_cube = os.listdir("data/train/cube/")
list_train_nocube = os.listdir("data/train/nocube/")
#list_train_cube=[1,2,3]
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
for i in enumerate(list_train_nocube):
    print i
    ip="data/train/nocube/"+i[1]
    img = cv2.imread(ip)

    img = cv2.resize(img,(150, 150))
    img = np.reshape(img,[1,3,150, 150])
    classes = model.predict_classes(img)

    img_res = cv2.imread(ip)
    cv2.putText(img_res,str(classes),(10,500), cv2.FONT_HERSHEY_SIMPLEX, 10,(255,0,0),10)
    #cv2.imwrite("result/train_nocube.png",img)
    cv2.imwrite("result1/train_nocube%d.jpg"%i[0],img_res)

#plt.imshow(img_res)
#print classes


# In[53]:





# In[70]:
