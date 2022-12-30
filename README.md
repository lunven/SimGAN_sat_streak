# Generate synthetic images from real fits images
The user needs to provide different parameters:
* --i: path to the fits images
* --o: beginning of the name given to the generated synthetic images, their targets and array indicating wether or not a streak is present in the image
  
  Example: three arrays npy will be created with this lign of command:
  * multi_test_samples.npy: generated synthetic images. Images without streaks are real patches.
  * multi_test_targets.npy: grountruths corresponding to the generated images
  * multi_test_patch_targets.npy: array of binary values indicating wether or not a streak is present in each image
```
python generate_synthetic.py --i "synthetic/" --o "multi"
```
