# CIFAR-100

CIFAR data sets are one of the most well-known data sets in computer vision. There are 100 different category labels containing 600 images for each (1 testing image for 5 training images per class). The 100 classes in the CIFAR-100 are grouped into 20 super-classes. Each image comes with a “fine” label (the class to which it belongs) and a “coarse” label (the super-class to which it belongs).

ReLUs()are not zero-centered activation functions and it leads to shift the bias term for the term in the next layer. But then, ELUs arranges the mean of the activation closer to zero because they have negative values, even if these values are very close to zero, and it converges faster, it means the model will learn faster.ELUs() have exponential term in the formula and the derivative of an exponential term, as you all know, equals to the exponential term itself. For the forward propagation, all weights and biases are activated with some constant multiplication of an exponential of them, and they are back-propagated with the derivative of the activation function, it is -actually- exponential of all weights and biases. 
    
    f(x) = { 0              if x >= 0                        f'(x) = { 0             if x >= 0
             a(exp(x)-1)    if x < 0                                   f(x) + a      if x < 0
             
             
Firstly, the cifar-100 dataset is loaded after importing the libraries.

Parameters are initialised.

Converting the labels in the data set into categorical matrix structure from 1-dim numpy array structure.

Normalising the images in the dataset.

Data Augmentation

One of the major reasons for overfitting is that you don’t have enough data to train your network. Apart from regularization,another very effective way to counter Overfitting is Data Augmentation. It is the process of artificially creating more images from the images you already have by changing the size, orientation etc of the image. It can be a tedious task but fortunately, this can be done in Keras using the ImageDataGenerator.Now, we will flow the data using our custom generator object for cropping the images. 
Additionally, the images are padded with four 0 pixels at all borders (2D zero padding layer at the top of the model).The model should be trained 32x32 random crops with random horizontal flipping. That’s all for data augmentation.
     
Building the model

    The CNN Architecture:
    18 convolutional layers arranged in stacks of
        (layers x units x receptive fields)
    ([1×384×3],
    [1×384×1,1×384×2,2×640×2],
    [1×640×1,3×768×2],
    [1×768×1,2×896×2],
    [1×896×3,2×1024×2],
    [1×1024×1,1×1152×2],
    [1×1152×1],[1×100×1]).
    Each stack has an ELU activation layer,MaxPooling layer and a dropout layer.
    Finally the layers are flattened and softmax activation is applied.
    
Compiling the model

Categorical cross-entropy has been picked as loss function since we have 100 category labels in the data set, and we already prepared the labels in the categorical matrix structure.Stochastic Gradient Descent with Momentum algorithm is used to optimize the weights on the back-propagation.Momentum term has been set to 0.9.
     
     
The model is evaluated on the test data and accuracy is detelmined.
      
    
