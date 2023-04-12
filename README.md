# DLAssignment2
Convolution Neural Networks.
# TO RUN TRAIN.PY

- train.py has the command Line arguments version of the code. Running it with appropriate parameters will result in training the network from scratch, from a pretrained model, or from a model entirely trained that needs to be loaded.

# Note :
- If wandb logging needed, kindly provide the key when prompted. Also, user may want to change the wandb project and entity names via the appropriate arguments.
- To load trained models, it must be made sure that a the model is in .pth format and is present in the same directory as the train.py file.
- System/ Environment should have pyTorch installed and updated, along with other basic libraries such as os, zipFile and requests.
- The train.py downloads the dataset(4 GBs) and extract its. Therefore around 30GBs of free space is recommended.
- The train.py will take care of whether to extract/ download. In case of a previous failed download, kindly deleted the failed attempt and run the code again.

# The Following are some example commandLine directives for running the train.py, user may copy paste the commands mentioned below

- python train.py --pretrain yes
- python train.py --load scratch.pth
- python train.py --log yes --pretrain yes

# To summarise the command Line arguments that can be passed to the train.py :
- 
