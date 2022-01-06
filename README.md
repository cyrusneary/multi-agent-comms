# Multiagent Systems that are Robust to Communication Loss

To run the two-agent navigation experiment from the paper, run
>python examples/ma_gridworld_total_corr.py

and to run the three-agent navigation experiment from the supplementary material, run

>python examples/run_three_agent_gridworld.py

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
