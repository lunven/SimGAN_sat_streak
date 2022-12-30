# Generate synthetic images from real fits images
The user needs to provide different parameters:
* --i: path to the fits images
* --o: beginning of the name given to the generated synthetic images, their targets and array indicating wether or not a streak is present in the image

```
python generate_synthetic.py --i "synthetic/" --o "multi"
