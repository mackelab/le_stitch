# Welcome

This package is written by: 

* Lars Buesing (primary contact), lars@stat.columbia.edu 

* Jakob Macke


This repository contains different methods for linear dynamical system models with Poisson observations. It has been developed and implemented with the goal of modelling spike-train recordings from neural populations, but at least some of the methods will be applicable more generally. 

In particular, the repository includes methods for

* Laplace approximation for state-inference
* Variational inference for the state
* Expectation maximisation for parameter learning, using Laplace or Variation inference
* Nuclear-norm minimisation (as described in Pfau et al.)
* Exponential family PCA
* Nonlinear Subspace Identification (SSID)
 
## Usage

To get started, run the example script: ./example/PLDSExample.m
or one of the other scripts in ./example


If you notice a bug, want to request a feature, or have a question or feedback, please make use of the issue-tracking capabilities of the repository. We love to hear from people using our code-- please send an email to info@mackelab.org.

The code is published under the GNU General Public License. The code is provided "as is" and has no warranty whatsoever. 

## Publications

The code is based on 

### Jakob H Macke, Lars Buesing, John P Cunningham, M Yu Byron, Krishna V Shenoy, and Maneesh Sahani. Empirical models of spiking in neural populations. In NIPS, pages 1350–1358, 2011.

[download paper](https://bitbucket.org/mackelab/pop_spike_dyn/downloads/Macke_Buesing_2012_Empirical.pdf)

### Lars Buesing, Jakob H Macke, and Maneesh Sahani. Spectral learning of linear dynamics from generalised-linear observations with application to neural population data. In NIPS, pages 1691–1699, 2012.

[download paper](https://bitbucket.org/mackelab/pop_spike_dyn/downloads/Buesing_Macke_2013_PLSID.pdf)

### David Pfau, Eftychios A Pnevmatikakis, and Liam Paninski. Robust learning of low-dimensional dynamics from large neural ensembles. In NIPS, pages 2391–2399, 2013.

and some of the methods are also described in 

### JH Macke, L Buesing, M Sahani: Estimating state and model parameters in state-space models of spike trains. Book-chapter, in preparation.

The code-package makes use of the optimisation-package minFunc, written by Mark Schmidt,
http://www.di.ens.fr/~mschmidt/Software/minFunc.html.