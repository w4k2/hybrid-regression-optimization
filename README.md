# Hybrid-Regression-Optimization

> The latest trends in computer networks bring new challenges and complex optimization problems, one of which is link dimensioning in {Spectrally-Spatially Flexible Optical Networks}. The time-consuming calculations related to determining the objective function representing the amount of accepted traffic require heuristics to search for good quality solutions. In this work, we address this problem by proposing a hybrid regression model capable of the objective function estimation. The presented algorithm uses a machine learning model built on already evaluated solutions for choosing new promising ones, providing a fast and effective method for solving the considered problem. The experimental evaluation conducted on two representative network topologies demonstrates that the proposed approach can significantly outperform other methods in the case of the EURO28 topology, while for the US26 topology, it provides results comparable to the solutions obtained so far.

## Running experiments


To replicate the experiment please follow this steps:

### 1. Install project requirements

To install libraries used in experiments please execute following command:

```python -m pip install -r requirements.txt```

### 2. Replicate experiments

Repository is provided with required datasets and results which were presented in article.

To replicate experiments please execute following scripts:

1. ```python e00_origins.py``` -- replicates experiment E1 (*Analysis of the distribution of the AT value in individual categories of solutions*). Result is stored in `figures` directory as `origins.png`.

2. ```python -W ignore regression.py``` -- replicates experiment E2 (*Assessment of the recognition model’s ability to distinguish between DSR and DSO members*) and E3 (*Assessment of the cross-predictive ability of the developed regression models regarding each category of labeled solutions.*). Results are printed as script output.


4. ```python -W ignore e01_main.py``` -- replicates experiment E4 (*Final evaluation of the effectiveness of the hybrid regression model in aiding link dimensioning*). Results are printed as script output.

All scripts were tested using python `python 3.8.12`.
