---
title: 'Introduction to Linear Regression'
date: 2022-02-18
permalink: /posts/2022/02/linear-regression/
excerpt_separator: <!--more-->
toc: true
tags:
  - stochastic
  - batch gradient
  - mini-batch gradient
---

This post provides an discussion on the Linear Regression algorithm and its implementation from scratch using Python.

<!--more-->

# Introduction
Linear Regression is a Supervised Learning algorithm used to solve problems where for every input(X), the respective outputs (Y) values are always discrete, and Logistic Regression is employed when output (Y) values are continuous. To understand Linear Regression, let us look into some real-world problems solved with this algorithm's help.

- Number of customers visiting the shop --> Profit for the shop owner
- Size of pizza --> Price of pizza
- Years of experience --> Salary of an employee
- Area of the house --> Price of the house


In all the examples, output/predicted values are continuous; hence Linear regression is used to build the machine learning model(hypothesis) and predict the output for a new set of inputs. To solve the problem of predicting an employee's salary based on various qualities of the employee with a Linear regression algorithm, we should apply the following steps:

 1. Data collection
 2. Model/hypothesis represenation
 3. Cost function
 4. Optimization

## 1. Data Collection

Let's consider/collect data records of salary provided by the company to emplyoees based upon the only single features of employees like years of experience. Since we are using single feature to predict salary of emplyoyee, we call this Univariate Linear regression.

<img src="/images/posts/linear-regression/picture1.png" alt="drawing" style="width:300px;"/>

Instead of a single feature, if we collect multiple features(qualities) of an employee to predict the salary, we call it multi-variate linear regression.

<img src="/images/posts/linear-regression/picture2.png" alt="drawing" style="width:550px;"/>

***Note:*** 
1. Number of samples range from 1, 2, 3, 4, . . . ., m
2. Number of features range from 0, 1, 2, 3, 4, . . . ., n where x0 is 1

<script src="https://gist.github.com/svgurlahosur/0dbea9d41815a0019270ee2b17990ce4.js"></script>


## 2. Model/hypothesis represenation

The model/hypothesis($$h_\theta(x)$$) approximates the target function by finding a relationship(pattern) between input and output using linear operators (since data is linear). We can define the hypothesis $$h_\theta(x)$$ as,


$$
h_\theta(x)=\ \theta_0x_0+\theta_1x_1\ +\ \theta_2x_2+\ \theta_3{x\theta}_3\ +\ \theta_4{x\theta}_4 \ + .\ .\ .\ \ \ \ +\ \theta_n{x}_n
$$

and we can rewrite the $$h_\theta(x)$$ as, $$ h_\theta\left(x\right)=\sum_{j=0}^{n}{\theta_jx_j} $$ 

where,
$$
_\ \theta_0,\ \theta_1,\theta_2,\theta_3,\theta_4,\ .\ .\ .\ .\ \theta_n $$ are model parameters and $$ 	x_0,\ x_1,x_2,x_3,x_4,\ .\ .\ .\ .x_n $$ are input features($$x_0$$ value is 1).

<script src="https://gist.github.com/svgurlahosur/34fa185718d44edb5823e57be0239329.js"></script>

The learning algorithm finds the optimal parameters in the hypothesis such that predicted outputs have a minimal error compared with actual output values using a cost function. Gradient descent is one such iterative optimization algorithm used to learn the Linear Regression model parameters. The gradient descent algorithm has three variations based on the number of samples considered for learning the model parameters.

**1. Stochastic gradient descent:** A single sample is used to predict model output, calculate the error, and optimize the model parameters.

**2. Batch gradient descent:** Batch of samples(m) are used to predict model output, calculate the error and optimize the model parameters.

**3. Mini batch gradient:** All the samples are used to predict model output, calculate error and optimize the model parameters.

**Note:** All the code snippets in the subsequent sections is for the Stochastic gradient descent algorithm and refer the [Batch gradient descent](https://github.com/svgurlahosur/Linear-regression/blob/main/Batch_Gradient_Descent.py) and [Mini batch gradient](https://github.com/svgurlahosur/Linear-regression/blob/main/Mini_Batch_Gradient_Descent.py) for respective implementaion.



## 3. Cost function

The cost function($$J(\theta)$$) is a metric to measure how well the hypothesis predicts outputs (Y) for a given input (X). It calculates the average squared difference (MSE) between the (MSE) between the predicted output and actual values. The goal of learning algorithm is to minimize the error by optimizing the model parameters with the Gradient descent algorithm.

$$
	J(\theta_0,\ \theta_1,\theta_2,\theta_3,\theta_4,\ .\ .\ .\ .\ \theta_n) = \frac{1}{2m}  \sum_{i=1}^{m} ({\mathrm{h_\theta}(x}^{(i)})\mathrm{-}y^{(i)}\mathrm{)^2}
$$


<script src="https://gist.github.com/svgurlahosur/fb8496165408c6a75e8b386e3a030621.js"></script>


## 4. Optimization

The cost function calculates the error for any given values model parameters($$\theta_0,\theta_1,\theta_2\ .\ .\theta_n$$), and the job of the optimization algorithm is to optimize these parameters for minimization of error. Hence, we can initialize the parameters with some initial values (hyperparameter initialization) and update them to minimize the error. The challenge here is to identify whether we should increase the value of parameters or decrease them. If we decide about the choice (increase/decrease), the next challenge is the magnitude by which we should update them.

Hence we can roughly imagine this problem where the model parameters are like vector elements with scope for optimization in both magnitude and direction. To calculate the direction of the update, we apply the gradient descent algorithm, and we use a hyperparameter called the learning rate($$ α $$) to calculate the magnitude of the update.

The gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. It calculates the partial derivative of the cost function($$J\left(\theta\right)$$) with respect to each model parameter to identify how the error changes as you tweak the parameters. After calculating gradients for each parameter, we update the parameter in the opposite direction since the gradient gives the direction of the steepest ascent. We use the learning rate to decide the magnitude of the update for each parameter. 

$$
\theta∶=\theta\ – α \frac{\partial}{\partial\theta}J\left(\theta\right) 
$$

Gradients for all the parameters are calculated with respect to the current cost before updating any parameters.

$$
\theta_j∶=\theta_j\ – α \frac{\partial}{\partial\theta_j}J\left(\theta\right)
$$

for all j = 0, 1, 2, 3, 4 , . . n 

Let us simplify the update rule for each parameter by calculating the paritial derivate of the cost function($$J(\theta)$$) with respect to each model parameter,


$$
\frac{\partial}{\partial\theta_j}J\left(\theta\right)=\frac{\partial}{\partial\theta_j}\frac{1}{2}\left(h_\theta\left(x\right)-y\right)\mathrm{2\ \ } 
$$

$$
=2\ .\frac{1}{2}\left(h_\theta\left(x\right)-y\right)\mathrm{\ }.\frac{\partial}{\partial\theta_j}\left(h_\theta\left(x\right)-y\right) 
$$

$$
=\left(h_\theta\left(x\right)-y\right)\mathrm{\ }.\frac{\partial}{\partial\theta_j}\left(\theta_0x_0+\theta_1x_1\ +\ \theta_2x_2+.\ .\ .\ .+\ \theta_nx_n-y\right)\mathrm{\ }
$$

$$
=\left(h_\theta\left(x\right)-y\right).x_j
$$

$$
\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$

using the update rule, we should update all $$\theta_j$$ where j = 0, 1, 2, 3, 4, . . . ., n


$$
\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)} \\

$$

$$
\theta_{1}:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{1}^{(i)} \\

$$

$$
\theta_{2}:=\theta_{2}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{2}^{(i)}
$$

..        ..        ..        ..
{: style="text-align: center;"}

..        ..        ..        ..
{: style="text-align: center;"}



$$
\theta_{n}:=\theta_{n}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{n}^{(i)}
$$


<script src="https://gist.github.com/svgurlahosur/df3f7cf28f07dcb88ccb41f7fa3617a5.js"></script>


Since the training data is exposed to the model during the training(parameter optimization) process, we use the testing data to verify how the model performs on unseen data after every epoch. 

<script src="https://gist.github.com/svgurlahosur/9dee2961d53f261b9e91bf148fe59654.js"></script>


The parameter optimization process is performed in an iterative loop until we reach minima since the gradient will be theoretically zero at that point(slope will be zero). The Hyperparameter Epochs define the number of times we perform the optimization process. As we progress with the optimization process with a predefined number of epochs, the error from the model starts decreasing. Meanwhile, we copy the model parameters and epoch number at which minimum error is obtained during the training process.

<script src="https://gist.github.com/svgurlahosur/8bc4b96a61ead238cd872be23905fb4f.js"></script>


The learning algorithm will be initialized with a predefined number of epochs and learning rate. Depending on the number of features, parameters are created, all initialized with value zero.  




<script src="https://gist.github.com/svgurlahosur/0543b0f601292723e6bf59a670679849.js"></script>


    ## [1] Epoch: 10 Training error: 3.7689470338877387 0.09422367584719346 Testing_error: 3.3261063075489092 Parameters: [1.4035586149969341, 9.789259471821223, 3.875418495881803, 8.638135375187337]
    ## [2] Epoch: 20 Training error: 2.0175238165739398 0.050438095414348495 Testing_error: 1.9598604430879836 Parameters: [1.3548566916116433, 9.715662275323327, 4.6655883649029395, 8.49529278286465]
    ## [3] Epoch: 30 Training error: 1.2902958925587842 0.03225739731396961 Testing_error: 1.4789717803567366 Parameters: [1.310219278149191, 9.622600723949207, 5.1742748799853215, 8.43168764311217]
    ## [4] Epoch: 40 Training error: 0.9915389458728094 0.024788473646820235 Testing_error: 1.2953993094257057 Parameters: [1.266895051995625, 9.560501390766953, 5.502509407456777, 8.393644390601194]
    ## [5] Epoch: 50 Training error: 0.8693544103073194 0.021733860257682987 Testing_error: 1.2271077013546419 Parameters: [1.2247045136869261, 9.52069800869814, 5.714321251775768, 8.370701194512849]
    ## [6]
    ## [7]
    ## [8]
    ## [9] -----------------------------------------------------Training Finished-----------------------------------------------------
    ## [10] 
    ## [11] Best error: 0.8693544103073194 at epoch: 50 with parameters values [1.2247045136869261, 9.52069800869814, 5.714321251775768, 8.370701194512849]
    ## [12]
    ## [13] ---------------------------------------------------------------------------------------------------------------------------


Graph plot to visualize how the training and testing errors are decresing as the model starts training.


<script src="https://gist.github.com/svgurlahosur/8391512a1056ceb4185eb4672dd20f29.js"></script>


<img src="/images/posts/linear-regression/picture3.png" alt="drawing" style="width:550px;"/>

The complete code for stochastic gradient descent algorithm can be found [here.](https://github.com/svgurlahosur/Linear-regression/blob/main/Stochastic_Gradient_Descent.py) 