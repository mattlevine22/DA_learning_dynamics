1. Set default shell to bash
   1. `chsh -s /usr/bin/bash levinema`
   2. Refresh terminal
2. Install conda (replace link with most recent)
   1. wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   2. bash Anaconda3-2023.09-0-Linux-x86_64.sh
   3. Refresh terminal
   4. conda update conda
   5. rm Anaconda3-2023.09-0-Linux-x86_64.sh
3. Set up base conda environment
   1. Install nvidia gpu visualization and github tool: `conda install -c conda-forge nvitop nvtop gh`
4. Set up github credentials using: https://stackoverflow.com/questions/71522167/git-always-ask-for-username-and-password-in-ubuntu
   1. gh auth login
   2. Select GitHub.com
   3. Select HTTPS
   4. Select Yes
   5. Select Paste an authentication token...to get an authentication token, follow these instructions: https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-to-create-a-GitHub-Personal-Access-Token-example
5. Set up MLDA environment:
   1. conda create --name mlda python=3.10
   2. conda activate mlda
   3. Install pytorch# (from [pytorch website](https://pytorch.org/get-started/locally/)
   -If on HPC with cuda 11.8: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
   -If on Mac with CPU only: `conda install pytorch torchvision -c pytorch`
   4. More installs:
   -`conda install matplotlib`
   -`conda install -c conda-forge pytorch-lightning `
   -`conda install -c conda-forge wandb `
   -`conda install -c conda-forge torchdiffeq`
   -`conda install -c conda-forge scikit-learn`
   -`conda install -c anaconda seaborn`
