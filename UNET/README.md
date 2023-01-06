## Description of the files:
* unet.py: contains the UNet and the different functions necessary to the training
* unet_training: file which needs to be run if the user wants to train the model. The model will be saved in the folder model. 
## Training the Unet
The user has to provide different parameters:
* --i name of the numpy array containing refined images
* --o name of the numpy array containing grountruth of the refined images
* --l learning rate 
* --v number of validation images
* --n number of epochs
* --k name given to the model
  
  Example of command lign: the unet will be trained for one epoch with a learning rate equal to 0.0001. After each epoch 500 images will be used to test the training. In the end of the training the model will be saved  under the name: "model.h5"
```
python unet_training.py --i "refined.npy" --o "target.npy" --l 1e-4 --v 500 --n 1 --k "model.h5"
```

