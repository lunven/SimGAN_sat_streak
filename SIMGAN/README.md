## Training the SIMGan
The user has to provide different parameters:

* --o beginning of the name given to synthetic images, their targets and the vector indicating the presence of streak in the synthetic image. This 
parameter has been chosen when these three numpy arrays have been created in 'generate_synthetic.py'. 
* --l learning rate 
* --b batch size
* --n number of epochs
  
  Example of command lign: the simgan will be trained for five epoch with a learning rate equal to 0.001 and with batch size equal to 24. 
```
python simgan_training.py  --o "multi" --b 24 --n 5  --l 0.001
```

## Generating refined images and their target from the refiner model

The user has to provide different parameters:

* --o beginning of the name given to synthetic images, their targets and the vector indicating the presence of streak in the synthetic image. This 
parameter has been chosen when these three numpy arrays have been created in 'generate_synthetic.py'
* --i name given to the refined images 
* --k name given to the targets of the refined images
* --m name of the refiner model used to improve the synthetic images
  
  Example of command lign: the simgan will be trained for five epoch with a learning rate equal to 0.001 and with batch size equal to 24. The refiner model and the discriminator model are saved under the name of "refiner.h5" and "discriminator.h5". The user can easily change their names in the code. 
```
python generate_refined_images.py --o "multi" --i "refined_test.npy" --k "target_test.npy" --m "refiner.h5"
```
