# Class project | [Insert title here]

This is our submission for the second project of "CS-439: Optimization for Machine Learning" where we conducted a study of the shape of the minima found by different optimizers, as well as how the curvature of the minima can affect generalization.

## Note on the used hardware
During the project, we made use of the free [Google Colab](https://colab.research.google.com) which permitted us to make use of a GPU (an NVIDIA Tesla K80 GPU, to be exact) as to make our code go faster, and to have more memory. Without a powerful GPU, the code may take a long time to execute, as we often train multiple image classification models.

## Problem description
In this project, we wanted to conduct a study on the kind of solutions first and second order methods of optimization, by looking at the landscape of the loss function around the found solution. That way, we would determine if it's a flat or sharp solution, and see how generalizable it is.

Our main motivation for this project was the work of [Dinh & al.](https://arxiv.org/abs/1703.04933), which suggests that sharp minima can indeed generalize well, and shouldn't be specifically avoided. 

The type of problem we chose is a classification one, using the MNIST dataset of hand-drawn digits.

## The 'src' folder
All code, may it be scripts or notebooks, are in the 'src' folder.

### Model
The model we used is the LeNet 5 architecture, a relatively simple convolutional neural network used for classification on 10 classes. It was implemented using the [Pytorch](https://pytorch.org/get-started/locally/) library, and the implementation can be found in the 'LeNet.py' file.

### Optimizers
To compare different optimizers for this project, we used three standard first order methods (Stochastic Gradient Descent, AdaGrad and Adam), and also a second order method optimizer called [AdaHessian](https://arxiv.org/abs/2006.00719). This state of the art method approximates second order information for its optimization, mainly the Hessian matrix's diagonal elements.

### Pyhessian


### Running the code