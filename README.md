# Multiagent Systems that are Robust to Communication Loss

This repository contains the code to run the numerical experiments from the paper:

Mustafa O. Karabag*, Cyrus Neary*, and Ufuk Topcu. 2022. Planning Not to Talk: Multiagent Systems that are Robust to Communication Loss. *In Proc. of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2022), Online, May 9–13, 2022*, IFAAMAS, 28 pages.

*Equal Contibution

[Read the paper on Arxiv](https://arxiv.org/pdf/2201.06619.pdf)

## Running the numerical experiments
To run the two-agent navigation experiment from the paper, execute
>python examples/ma_gridworld_total_corr_add_end_state.py

and to run the three-agent navigation experiment from the supplementary material, execute

>python examples/run_three_agent_gridworld_add_end_state.py

## Requirements
This project requires the following Python 3 packages:
- Numpy
- matplotlib
- cvxpy
- Mosek

### Installation instructions using Anaconda
- Download and install Anaconda from (https://www.anaconda.com/products/individual-d).
- To create a new virtual environment run: 
  - >conda create -n mac python=3.9
- Activate the virtual environment:
  - >conda activate mac
- Install the necessary packages:
  - >conda install numpy
  - >conda install matplotlib
- Follow the OS-specific installation instructions for cvxpy (https://www.cvxpy.org/install/) and Mosek (https://docs.mosek.com/9.2/pythonapi/install-interface.html).
