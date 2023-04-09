# Vision-spynet

implementation of SpyNet in PyTorch for Computer Vision course project

## What SpyNet does?

This project addresses optical flow estimation in a fused approach that employs image pyramids and convolutional neural networks. Using image pyramids enables the method to estimate optical flow in a coarse-to-fine strategy. 
At each level of the pyramid, the algorithm estimates motions by warping one image of a pair at each pyramid level by the current flow estimate obtained from a CNN and then updates the estimation for the next level. Consequently, Unlike FlowNet, the networks do not need to cope with large motions.

## Inference 
This is a Python script for performing inference using a pretrained SpyNet model for optical flow estimation. The `inference.py` script takes input frames from a dataset, computes optical flow using the SpyNet model, and visualizes the results.

## Requirements
- Python 3.8
- PyTorch > 2.0.0
- NumPy > 1.23.5
- Matplotlib > 3.7
- cv2 > 4.7
- tensorboard > 2.12

## Usage
1. Install the required dependencies using `pip` or `conda`.
2. Download the pretrained SpyNet model checkpoint and place it in the appropriate directory.
3. Modify the `data_root` and `checkpoint_name` arguments in the `inference()` function call to specify the location of your dataset and the path to the pretrained model checkpoint, respectively.
4. Optionally, set `show_acc` to `True` if you want to display the accuracy of the optical flow estimation.
5. Run the script using `python inference.py` in your terminal or Python environment.
6. The script will generate visualizations of the optical flow estimation results, including an image plot and a quiver plot.

Note: The script assumes that the input frames are in the Monkaa_cleanpass dataset format, but can be easily modified to support other datasets by changing the `valid_ds` dataset instantiation to the appropriate dataset class and providing the correct data root.

## References
- [SpyNet with Pytorch](https://github.com/Guillem96/spynet-pytorch)
- [Monkaa and Driving_cleanpass dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Original SpyNet implementation](https://github.com/anuragranj/spynet)
