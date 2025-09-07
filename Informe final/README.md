# Deep reinforcement learning

Deep reinforcement learning could be seen as a subfield of machine learning which combines deep learning techniques 
and reinforcement learning ones, and it differs from common machine learning approaches because:

1. We have not a labeled dataset, just a reward signal which only will be discovered through environment interaction
2. We have many function compositions which form the objective function, not just a convex error function.

In order to close those gaps we have some heuristics, like replay buffer.

# Project structure

We have a bunch of experiments here, where each experiment is composed by:

```txt 
agents 
    agent_x.py
environments
    environment 
cmd
    cli.py
```

The cli.py is very important here because allow users to use the main implementation

