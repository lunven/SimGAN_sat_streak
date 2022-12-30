## Training the Unet
The user has to provide different parameters:
* --i name of the numpy array containing refined images
* --o name of the numpy array containing grountruth of the refined images
* --l learning rate 
* --v number of validation images
* --n number of epochs
Example of command lign: the unet will be trained for one epoch with a learning rate equal to 0.0001. After each epoch 500 images will be used to test the training.
```
python unet_training.py --i "refined_l=1e-6small_.npy" --o "target_l=1e-6_.npy" --l 1e-4 --v 500 --n 1
```
