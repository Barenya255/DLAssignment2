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

## The Following are some example commandLine directives for running the train.py, user may copy paste the commands mentioned below

- python train.py --pretrain yes
- python train.py --load scratch.pth
- python train.py --log yes --pretrain yes

## To summarise the command Line arguments that can be passed to the train.py :
- shorthand: '-lg', argument: '--log', type = str, default = "no"
- shorthand: '-ld', argument: '--load', type = str, default = "NIL", help = "enter name of model to load into and place the model in the same directory."
- shorthand: '-wp', argument: '--wandb_project', type = str, default = "DLAssignment2", help = "Default has been set to my project name. Please change as per required"
- shorthand: '-we', argument: '--wandb_entity', type = str, default = "cs22m028"
- shorthand: '-e',, argument:'--epochs',type = int, default = 20
- shorthand: '-b',, argument:'--batch_size',type = int, default = 32
- shorthand: '-f',, argument:'--factor',type = int, default = 2
- shorthand: '-lr', argument: '--learningRate', type = float, default = 1e-4
- shorthand: '-w_d, argument:, '--weight_decay', type = float, default = 0
- shorthand: '-d',, argument:'--dropOut', type = float, default = 0.5
- shorthand: '-k',, argument:'--kernels', type = str, default = "5 5 3 3 3", help = "Enter 5 space separated integers")
- shorthand: '-fs', argument: '--filterSize', type = int, default = 32, help = "Enter 5 space separated integers"
- shorthand: '-a',, argument:'--activation', type = str, default = "mish", help = "can choose from only mish, silu, gelu and relu"
- shorthand: '-pre, argument:, '--pretrained', type = str, default = "no", help = "train from scratch or from pretrained weights"
- shorthand: '-aug, argument:, '--augmentation', type = str, default = "yes", help = " augment data or not"
- shorthand: '-bN', argument: '--batchNorm', type = str, default = "yes", help = " augment data or not"

