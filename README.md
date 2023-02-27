# Motion Estimation 
This tutorial covers some algorithms for motion estimation used in video compression. 

## Preparation for CPP users
0. if your computer has not installed opencv before, install opencv like this following command 
```
cd comp5425_assignment2_motion_estimation \
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip \ # you can manually download the source code of opencv
unzip opencv.zip \
mkdir build && cd build \
cmake .. -DBUILD_LIST=core,imgcodecs,highgui \  # only build these modules
make -j && \
echo "build opencv with opencv_core, opencv_imgcodecs !"
```

1. build the project
```
cd comp5425_assignment2_motion_estimation \
mkdir build && cd build \
cmake .. \
make -j \
```

Usage example:
```
./build/motion_estimation -r ./data/00013.jpg -c ./data/00014.jpg -b 16 -s 15
```
this will do motion estimation with block size 16 and search range [-15, 15] for reference frame `./data/00013.jpg` and current frame `./data/00014.jpg`.

**TASK**: you are required to implement the function *motion_estimation_2dlog_search* to achieve a 2D logarithmic search for motion estimation. 

## Preparation for Python users
0. install dependencies 
```
pip3 install -r requiresments.txt:
```
**TASK** : you are required to implement the function *motion_estimation_2dlog_search* in *python/motion_estimation.py*


## Reference
- COMP5425, HK PolyU (lecturer: Prof. Changwen Chen)
