# DLAssignment2
Convolution Neural Networks.

# For Notebooks of PARTA and PARTB
- user neeeds to make sure that all dependent and imported libraries are installed in the environment
- colab does not provide wandb by default and hence a pip install is necessary.

# TO RUN TRAIN.PY

- train.py has the command Line arguments version of the code. Running it with appropriate parameters will result in training the network from scratch, from a pretrained model, or from a model entirely trained that needs to be loaded.

# Note :
- kernels will have to passed as strings containing space separated integers eg : "5 3 3 3 3"
- If wandb logging needed, kindly provide the key when prompted. Also, user may want to change the wandb project and entity names via the appropriate arguments.
- To load trained models, it must be made sure that a the model is in .pth format and is present in the same directory as the train.py file.
- System/ Environment should have pyTorch installed and updated, along with other basic libraries such as os, zipFile and requests.
- The train.py downloads the dataset(4 GBs) and extract its. Therefore around 30GBs of free space is recommended.
- The train.py will take care of whether to extract/ download. In case of a previous failed download, kindly deleted the failed attempt and run the code again.

## The Following are some example commandLine directives for running the train.py, user may copy paste the commands mentioned below
- python train.py --log yes -lr 0.01 -b 64 -e 2 -wp Testing2 -we cs22m028 -f 1 -w_d 0.00001 -d 0.5 -k "5 3 3 3 3" -fs 64 -pre yes -bN no
- python train.py --pretrain yes
- python train.py --load scratch.pth
- python train.py --log yes --pretrain yes

## To summarise the command Line arguments that can be passed to the train.py :
- shorthand: '-lg', argument: '--log',             Desription: set to no if logging not desired, otherwise, wandb logging done. Prompt to insert key will appear                 
- shorthand: '-ld', argument: '--load',            Description: set to no if do not wish to load a saved model, otherwise saved model's path(relative to current directory) must be passed.           
- shorthand: '-wp', argument: '--wandb_project'    Description: set wandb project name. Default to DLAssignment2
- shorthand: '-we', argument: '--wandb_entity'     Description: set wandb entity name. Default to cs22m028. 
- shorthand: '-e',, argument:'--epochs',           Description: set number of epochs for which to train the model (default = 10)
- shorthand: '-b',, argument:'--batch_size',       Description: set batch size ( default = 32 )   
- shorthand: '-f',, argument:'--factor',           Description: set factor by which to increase the number of filters ( default = 32 )
- shorthand: '-lr', argument: '--learningRate',    Description: set learning Rate (default to 0.0001)
- shorthand: '-w_d, argument:, '--weight_decay',   Description: set weight decay ( default = 0 )
- shorthand: '-d',, argument:'--dropOut',          Description: set dropOut probability ( default = 0.5 )
- shorthand: '-k',, argument:'--kernels',          Description: set kernel sizees, enter a string of 5 integers ( 5 5 3 3 3)
- shorthand: '-fs', argument: '--filterSize',      Description filter size: number of channels
- shorthand: '-a',, argument:'--activation',       Description: set activation function ( default = "mish" )
- shorthand: '-pre, argument:, '--pretrained',     Description: set whether to load vision transformer model or not ( default = no )
- shorthand: '-aug, argument:, '--augmentation',   Description: set whether to augment dataset or not ( default = yes )
- shorthand: '-bN', argument: '--batchNorm',       Description: set whether to use batchNorm or not ( default = yes )

## Note:
- all parameters except for wandb project/entity names are to be kept to lower case.
- preTrained VisionTransformer model could not be added to the GitHub repository as it is huge (> 700 MBs)

