## Training the Unet
The user has to provide different parameters:
* --i name of the numpy array containing refined images
* --o name of the numpy array containing grountruth of the refined images
* --l learning rate 
* --v number of validation images
* --n number of epochs
```
python unet_training.py --i "refined_l=1e-6small_.npy" --o "target_l=1e-6_.npy" --l 1e-4 --v 500 --n 1
```
