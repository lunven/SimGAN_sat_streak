## Generating realistic synthetic streaks
With the number of space debris growing day by day, it is essential to be able to monitor them to ensure safety on earth and in space. 
What is proposed in this project is an automated method to improve the reality of synthetic satellite streaks  in astronomical images using a deep learning method: SimGAN. It is a type of generative adversarial network, specifically designed for image-to-image translation tasks. For this, pictures of space taken with an OmegaCAM camera on the VLT telescope in Chile are used.  The efficiency of the refinement is tested by comparing the segmentations obtained with a UNet trained on synthetic images and one trained with synthetic images refined by the SimGAN model. Unfortunately, no clear improvements are observed.
<p align="center">
  <img src="images/real_image.png" length="20" >
  
</p>

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
# Get patches of real streaks 
A csv file where it is written if an image contains a streak or not and if yes the coordinates of the two extreme points of the streak. 
The user needs to provide:
* --i: path to the png images containing real satellite streaks
```
python simgan_training.py --i "images/" 
```
