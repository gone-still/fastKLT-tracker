# fastKLT-tracker
Python port of [Zana Zakaryaie's FAST-KLT tracker](https://github.com/zanazakaryaie/object_tracking/tree/main/fast_klt) originally written in C++. 
I'm still working on the implementation. This is the tracker's current output:


https://user-images.githubusercontent.com/8327505/152904250-2de1e28c-1ecc-4706-a3f8-384376df6488.mp4


The algorithm first uses a caffe deep model for the initial face detection. It then extracts **FAST** keypoints from this ROI and tracks them using **KLT**. The new keypoints are used to calculate a new bounding rectangle on the new frame. The result is one initial detection and a fast, smooth (and somewhat robust to occlusion) tracking for new frames. Full details [here](http://imrid.net/?p=4441).
