# FinIto
PhD research devoted to feature engineering and forecasting of stochastic processes. Ito processes are regarded particularly.

Prime goal of the research are robust, innovative forecasting algorithms.
Purpose of this repository it to wrap all reasonable approaches into python
module.

# Installation

Install necessary dependencies:
`pdm add`

Install additional dependencies:
```
pdm add -G notebooks  # For launching notebooks
pdm add -G test  # For unit-testing
pdm add -G dev  # If you want to join development
```

# Theory

Ito's lemma is the key point of estimating derivatives prices.

# Fields of study

We can consider coefficients in Ito's equation as a random functions. It 
lets us estimating distributions of them and use estimated values as a 
features. 
To do so we can solve optimization problem on empirical and theoretical 
distribution. 
This vector of research opens multiple sub-problems:
- Mesh methods
- Loss-functions
- View of theoretical distribution

Or we can develop approach, that doesn't requires specific nature of the 
stochastic process. We gave it a name - "Taylor modification".
