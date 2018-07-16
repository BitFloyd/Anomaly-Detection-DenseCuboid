import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import numpy as np
from skimage import data
from skimage.transform import resize


def convolution2d(image, kernel, bias):
    m, n , c = kernel.shape
    if (m == n):
        y, x ,c = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m,:]*kernel) + bias
    return new_image


model = VGG16(weights='imagenet', include_top=False)
model.summary()


conv1 = model.get_layer(name='block1_conv1')

conv1_filters = conv1.get_weights()[0]


conv1_filters_intact = conv1.get_weights()[0]
conv1_filters_bias = conv1.get_weights()[1]


conv1_filters = conv1_filters - np.min(conv1_filters)
conv1_filters = conv1_filters/(0.0+np.max(conv1_filters))

new_array_filters = np.zeros(shape=(5,5,3,64))

new_array_filters[1:4,1:4,:,:] = conv1_filters

image_filters = np.zeros(shape=(40,40,3))

for i in range(0,8):
    for j in range(0,8):
        image_filters[i*5:(i+1)*5,(j*5):(j+1)*5,:] = new_array_filters[:,:,:,i*8+j]


imgplot = plt.imshow(image_filters)
plt.axis('off')
plt.savefig('convolutional_filters.pdf',bbox_inches='tight')
plt.close()



test_image = data.astronaut()

size = 128

test_image = resize(test_image,output_shape=(size,size))

imgplot = plt.imshow(test_image)
plt.axis('off')
plt.savefig('test_image.pdf',bbox_inches='tight')
plt.close()

list_images = np.zeros(shape=(size,size,64))

for i in range(0,64):
    list_images[1:size-1,1:size-1,i]=convolution2d(test_image,conv1_filters_intact[:,:,:,i],conv1_filters_bias[i])

big_image_array = np.zeros(shape=(size*8,size*8))

for i in range(0,8):
    for j in range(0,8):
        img = list_images[:,:,i * 8 + j]

        img = img-np.min(img)
        img = img/np.max(img)

        big_image_array[i * size:(i + 1) * size, (j * size):(j + 1) * size] = img


imgplot = plt.imshow(big_image_array,cmap='gist_gray')
plt.axis('off')
plt.savefig('convolutional_outputs_small.pdf',bbox_inches='tight')
plt.close()