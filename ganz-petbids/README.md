# PET BIDS tutorial

In this tutorial, I will introduce you quickly to the PET BIDS standard and it's applications. The tutorial will consist of three parts:

1) First, I will introduce you quickly to the PET BIDS format
2) Then, I will show you how to convert a positron emission tomography dataset into PET BIDS format using the [PET2BIDS](https://github.com/openneuropet/PET2BIDS) converter. A much more extended version of the conversion part of this this tutorial can be found in the [BIDS Starter Kit](https://bids-standard.github.io/bids-starter-kit/).
3) Finally, we will run a single PET analysis (a simple kinetic modelling analysis) with a simple processing pipeline developed based on PETSurfer. A more extensive description of the tutorial is given in the [PET_pipelines](https://github.com/openneuropet/PET_pipelines/tree/main/pyPetSurfer) repository.

## Preparations

There are three steps to prepare for the tutorial:

1. For the conversion part, download this small example dataset [here](https://drive.google.com/file/d/10S0H7HAnMmxHNpZLlifR14ykIuiXcBAD/view?usp=sharing).

2. For the data processing part, download this example dataset from OpenNeuro: https://openneuro.org/datasets/ds001421

3. Set up the environment and add the required python packages by opening the jupyter notebook called "PET_BIDS_tutorial.ipynb" and completing **Getting started with the Python environment and packages**.



## Acknowledgements
Most of these tools were developed by collaborators within the [OpenNeuroPET](https://openneuropet.github.io/) project. OpenNeuroPET is a collaboration between Stanford university, NIH, MGH  and the Neurobiology Research Unit (NRU) at Copenhagen University  Hospital. It is funded through the BRAIN initiative and  the Novo Nordisk foundation. 

