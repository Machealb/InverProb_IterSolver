# InverProb_IterSolver

<img src="figs/InverseProblem.png" width="500" />


* InverProb_IterSolver is a MATLAB code library for solving linear inverse problems with regularization.

1. The first goal is to provide subspace projection based iterative methods than can solve
large-scale discrete linear inverse problems with regularization

$$\min_{x\in\mathbb{R}^{n}}\{\|Ax-b\|_{2}^{2}+\lambda R(x)\},$$

where $R(x)$ is the regularizer. It can be either a linear Tikhonov regularization term or  a nonlinear regularization term such as $L_p$ or TV regularization.

2. The second goal is the provided high-scalable iterative methods for Bayesian inverse problems, with untilities for efficient uncertainty quantification and sampling.


## Notice
At this stage, this code should be used with the following two famous Inverse Problems solver packages:

[1]. P. C. Hansen, Regularization Tools version 4.0 for Matlab 7.3, Numer. Algor., 46 (2007), pp. 189-194.

[2]. S. Gazzola, P. C. Hansen, and J. G. Nagy, IR Tools: A MATLAB package of iterative regularization methods and large-scale test problems, Numer. Algor., 81 (2019), pp. 773-811.


## Submit an issue
You are welcome to submit an issue for any questions related to InverProb_IterSolver. 


## Here are some research papers using InverProb_IterSolver
1. Haibo Li. "[A preconditioned Krylov subspace method for linear inverse problems with general-form Tikhonov regularization. SIAM Journal on Scientific Computing, 46(4), A2607–A2633.](https://doi.org/10.1137/23M1593802)."
2. Haibo Li. "[Subspace projection regularization for large-scale Bayesian linear inverse problems](https://arxiv.org/abs/2310.18618)."
3. Haibo Li. "[Projected Newton method for large-scale Bayesian linear inverse problems](https://arxiv.org/abs/2403.01920)."
## License
If you use this code in any future publications, please cite like this:

Haibo Li. "[A preconditioned Krylov subspace method for linear inverse problems with general-form Tikhonov regularization. SIAM Journal on Scientific Computing, 46(4), A2607–A2633.](https://doi.org/10.1137/23M1593802)."
