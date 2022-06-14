# Class project | [Insert title here]

This is our submission for the second project of "CS-439: Optimization for Machine Learning" where we conducted a study of the shape of the minima found by different optimizers, as well as how the curvature of the minima can affect generalization.)

## Note on the used hardware
During the project, we made use of the free [Google Colab](https://colab.research.google.com) which permitted us to make use of a GPU (an NVIDIA Tesla K80 GPU, to be exact) as to make our code go faster, and to have more memory. Without a powerful GPU, the code may take a long time to execute, as we often train multiple image classification models.

## Problem description
In this project, we wanted to conduct a study on the kind of solutions first and second order methods of optimization, by looking at the landscape of the loss function around the found solution. That way, we would determine if it's a flat or sharp solution, and see how generalizable it is.

Our main motivation for this project was the work of [Dinh & al.](https://arxiv.org/abs/1703.04933), which suggests that sharp minima can indeed generalize well, and shouldn't be specifically avoided.

## The 'src' folder

### Model