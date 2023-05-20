# Forward-Forward algorithm in PyTorch - Exploration

## Current Examples

Below is the instructions for running the first pass on experimenting with the forward-forward algorithm.

```bash
git clone https://github.com/tosterberg/forward-forward-exploration.git
cd forward-forward-exploration
python3 -m main
```

## TODO's

### Discussion
- positive vs negative pass (real vs. not real)
  - aside: boltzmann machine positive and negative phase corollary?
- goodness function (what is it doing really)
  - immediate weight updates (how to calculate)
- negative data generation (what makes good negative data)
  - aside: boltzmann machine goes to equilibrium during the negative phase
  - naive: mis-label the data
  - paper: mask two real inputs to generate two fake outputs

### Implementations
- forward-forward layer implementation
- goodness function implementations
- forward-forward negative data generators
- train using
  - MNIST
  - CIFAR
  - others?

### Experiments
- compare forward-forward to backprop
  - create baseline network for testing
  - test hyperparameters?
  - test different models?
- compare goodness functions
- compare negative data generation methods
- compare alternate paths for goodness vs negative data (sign flip)
  - compare the conjunction of goodness and activation function
  - compare reversing the paths, minimizing activation on positive data and maximizing activation on negative data
- compare ratios of positive forward passes to negative forward passes and effect on training

## References

- Paper: [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/pdf/2212.13345.pdf)