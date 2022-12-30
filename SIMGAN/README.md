## Training the SIMGan
The user has to provide different parameters:

* --o beginning of the name given to synthetic images, their targets and the vector indicating the presence of streak in the synthetic image. This 
parameter has been chosen when these three numpy arrays have been created in 'generate_synthetic.py'
* --l learning rate 
* --b batch size
* --n number of epochs
  
  Example of command lign: the simgan will be trained for five epoch with a learning rate equal to 0.001 and with batch size equal to 24. 
```
#python simgan_training.py  --o "multi" --b 24 --n 5  --l 0.001
```
