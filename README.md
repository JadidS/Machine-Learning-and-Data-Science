# Machine-Learning-and-Data-Science

Here is a list of some Machine Learning and Data Science projects I have completed.

They have all been deployed with Amazon Sagemaker, ECR, ECS, and Docker containers in AWS

More detailed explanations are available with the code.


MNIST CLASSIFIER:

This project consists of a supervised learning MNIST digit classifier using Scikit-Learn and Numpy.

We use start by using a voting classifier with Random Forest Classifier, Extra Trees Classifier, Linear Support Vector Classifier, and Multilayer Perceptron Classifier.

We compare this result with a Stacking Ensemble with a blender.

We get an accuracy score of 0.9723 with Extra_Trees, 0.9657 with voting classifier, and 0.9682 with our Stacking Ensemble.


CLUSTERING/PREPROCESSING MAKE_MOONS:

This project consists of an unsupervised learning clustering/preprocessing of the make_moons dataset using Scikit-Learn, Numpy, and Matplotlib.

We use the DBSCAN algorithm to cluster the data. We also use Kneighbors classifier to classify new instances.

We then try using Spectral Clustering and the Bayesian Gaussian Mixture Model.

DBSCAN, Kneighbors, and Spectral Clustering gave us very good results with the right hyperparameters but the Gaussian Mixture tried too hard to make our clusters ellipsoids and didn't cluster as well as our other models.


GENERATING FASHION MNIST PICTURES WITH DCGANs:

This project consists of generating grayscale images with a Deep Convolutional GANs using Tensorflow, Keras, and Numpy.

We make a generator that takes random Gaussian distribution inputs and a discriminator which takes real images from training set or generator as input and must guess if real or fake.

We try the same data with hashing using a Binary Autoencoder which uses an encoder to output our 28x28 images as a vector size of 16 and a decoder which take the input of size 16 and outputs 28x28 arrays.

In both cases outputs are a bit blurry and we should train our model a bit more and add layers to both the encoder and decoder. If the model becomes too strong, the outputs would be better but the model wouldn't learn the useful patterns of the data.


GENERATING SHAKESPREARE TEXT WITH NLP RNN:

The project consists of creating fake Shakespeare text with Natural Language Processing using a Stateful RNN using Tensorflow, Keras, and Numpy.

We use Kera's tokenizer class and create dataset windows to preprocess the data before training the model to predict the next character using the previous 100 characters.

We got an output of: from most breathe life unto him, with care.

We finally have some Shakespeare text!

It could be better but still quite good.
