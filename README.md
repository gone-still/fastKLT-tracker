# fastKLT-tracker
Python port of [Zana Zakaryaie's FAST-KLT tracker](https://github.com/zanazakaryaie/object_tracking/tree/main/fast_klt) originally written in C++. 
I'm still working on the implementation. This is the tracker's current output:

https://user-images.githubusercontent.com/8327505/148007760-1e32f528-9ec5-4757-8f3d-b88bf81ce6bd.mp4

The algorithm first uses a caffee deep model for the initial face detection. It then extracts **FAST** keypoints from this ROI and tracks them using **KLT**. The new keypoints are used to calculate a new bounding rectangle. The result is an initial detection and a fast, smooth tracking. Full details [here](http://imrid.net/?p=4441).
