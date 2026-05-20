# Deep Learning from Scratch — Learning Notebooks

This repo is my working-through of Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
series. It exists to build (and document) my own understanding of how
neural networks and backpropagation work at the lowest level.

It is a **learning log, not a research project.** I keep intermediate
steps, mistakes, and dead ends in the notebooks on purpose — the point is
to show the reasoning, not just a clean final result.

## Contents

- `micrograd_from_scratch.ipynb` — a scalar-valued autograd engine and a
  small MLP, built up from the definition of a derivative. Covers the
  `Value` class, the computation graph, `_backward` for each operation,
  topological-sort backprop, the gradient-accumulation fix for reused
  nodes, and a basic training loop.

*(Planned: makemore, then the GPT notebooks. This list will grow as I
work through the series.)*

## What I'm doing here

Each notebook is annotated in my own words: derivations, why a step
works, and bugs I hit along the way (e.g. gradients overwriting instead
of accumulating when a node feeds into itself). Where it helped, I
cross-checked results against PyTorch's autograd.

## What this is not

These are reimplementations of well-known educational material — the
ideas and structure are Karpathy's. This repo demonstrates that I dig
into fundamentals carefully; it is not original work, and it is a
starting point for the harder projects it's preparing me for.

## Background

Coming into ML from a quantitative/auditing background, currently doing
an M.Sc. in Analytics & AI. Longer-term interest: mechanistic
interpretability and alignment of frontier models.