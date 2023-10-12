This repository is derived from work presented in [Levine and Stuart 2022](https://arxiv.org/abs/2107.06658). To cite this repo, please use:
```
@article{levine2022framework,
  title={A framework for machine learning of model error in dynamical systems},
  author={Levine, Matthew and Stuart, Andrew},
  journal={Communications of the American Mathematical Society},
  volume={2},
  number={07},
  pages={283--344},
  year={2022}
}
```

## Experiments

1. Lorenz 63 with partial noisy observations (only first component), see `run_scripts/basic_l63`
   1. To jointly learn DA hyperparameters and NN-based right-hand-side for an ODE, see:  `run.py`
   2. To optimize only DA hyperparameters (using true known equations of l63), see: `run_3dvar_trueODE_learnK.py`
2. [Work in progress] Earth magnetic pole model with partial noisy observations (only dipole observed)---see `run_scripts/earth_dipole_g12`
   1. To see that simple 3dvar data assimilation  works for this setting on short intervals with sample rate 0.1 and light noise (using true known model), see: `run_3dvar.py`
   2. To learn NN-based right-hand-side for an ODE, see:  `run_learnNN_moreData.py` and `run_learnNN.py`.
3. Lorenz 96 Multi-scale system---not yet built in this repository, but some results are in Levine and Stuart 2022.
   1. In this system, it would be especially impressive to watch the necessity of learning hidden dynamics increase proportionally with the lack of scale separation encoded by $\varepsilon$.
4. Fermi-Pasta-Ulam-Tsingou---for future work!
   1. Houman Owhadi has suggested this system as another good benchmark
   2. See eq. 6.7 in [Tao, Owhadi, and Marsden 2010](https://arxiv.org/abs/0908.1241), and observe only the midpoint of the stiff springs.
   3. Perhaps consider learning a Hamiltonian and building an autodifferentiable symplectic Euler scheme.

Note: when I run locally on a laptop, I set `accelerator='cpu'`:
- `python run_learnNN.py --accelerator cpu`

Also: when I debug or want to make sure things are installed correctly and run, I use `fast_dev_run=True`:
- `python run_learnNN.py --fast_dev_run True`

## Next steps
Note: If any of these interest you, please reach out to me and we can work together! I am very much looking for collaborators to help tackle these challenging questions.

0. Test in new challenging settings
   - Systems with real data!
   - Relevant toy models that capture key challenges in working with real data
1. Implement other auto-differentiable data assimilation schemes
   1. EnKF: paper by [Chen, Sanz-Alonso, and Willet 2022](https://arxiv.org/abs/2107.07687)) and [code](https://github.com/ymchen0/torchEnKF#auto-differentiable-ensemble-kalman-filters-ad-enkf)
   2. Continuous DA: the key idea is to first interpolate the observations, then solve an ODE/SDE that is forced by differences between the differentiated interpolant and observables of the solution.
      - Continuous-time 3DVAR: see eq. 14 in [(Blömker, Law, Stuart, and Zygalakis)](https://arxiv.org/abs/1210.1594)
      - Continuous-time EnKF [(Calvello, Reich and Stuart 2022)](https://arxiv.org/abs/2209.11371)
2. Learning Stochastic Differential Equations
   - An idea for this is to learn parameters for the stochastic forcing, and sample multiple i.i.d. solution paths at each mini-batch of SGD
   - Can we learn to correctly distinguish the amount/structure of noise present in the dynamics vs in the observation?
3. Learning more interpretable models
   - Test sparse dictionary learning by simply enumerating a library and penalizing coefficients with L1.
   - Try to encourage low-rank structure in hidden dynamics so that only the minimal number of hidden states is active (i.e., for L63, I want to hypothesize 10 hidden dimensions, and automatically nullify 7 of them).
4. Identifying efficient architectures
   - e.g., does putting a scale factor in front of each right-hand-side improve learning (NN's like their outputs to be around [-1,1])?
   - Can we use this to improve hybrid modeling settings where the RHS is actually rather large (e.g. 100)?
5. Improving objective functions
   - How can we leverage non-temporal physical knowledge/data (e.g., invariant statistics) to further improve our learning?
   - Can we penalize difference between moments of the predicted vs known invariant distribution? Can this be autodifferentiated efficiently?

## Setting up server and virtual environment
1. Set default shell to bash (substitute your username...or don't do this if you don't need to)
   1. `chsh -s /usr/bin/bash levinema`
   2. Refresh terminal
2. Install conda (replace link with most recent)
   1. wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   2. bash Anaconda3-2023.09-0-Linux-x86_64.sh
   3. Refresh terminal
   4. conda update conda
   5. rm Anaconda3-2023.09-0-Linux-x86_64.sh
3. Set up base conda environment
   1. Install nvidia gpu visualization and github tool: 
      - `conda install -c conda-forge nvitop nvtop gh`
4. Set up github credentials using: https://stackoverflow.com/questions/71522167/git-always-ask-for-username-and-password-in-ubuntu
   1. gh auth login
   2. Select GitHub.com
   3. Select HTTPS
   4. Select Yes
   5. Select Paste an authentication token...to get an authentication token, follow these instructions: https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-to-create-a-GitHub-Personal-Access-Token-example
5. Set up MLDA environment:
   1. `conda create --name mlda python=3.10`
   2. `conda activate mlda`
   3. Install pytorch# (from [pytorch website](https://pytorch.org/get-started/locally/)
   - If on HPC with cuda 11.8: 
     - `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
   - If on Mac with CPU only: 
     - `conda install pytorch torchvision -c pytorch`
   1. More installs:
   - `conda install matplotlib`
   - `conda install -c conda-forge pytorch-lightning`
   - `conda install -c conda-forge wandb`
   - `conda install -c conda-forge torchdiffeq`
   - `conda install -c conda-forge scikit-learn`
   - `conda install -c anaconda seaborn`
