# SQPnP
OpenCV C++ Implementation of the ECCV Paper: "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem"

```
@inproceedings{terzakis2020SQPnP,
title={A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem},
author={George Terzakis and Manolis Lourakis},
booktitle={European Conference on Computer Vision},
pages={},
year={2020},
publisher={Springer}
}
```

As of 10/1/2020, outperforms author's implementation by approximately 50us.

## Dependencies
- OpenCV >= 3.4
- Eigen >= 3 (for timing comparison)

## Building
```
mkdir build && cd build && cmake ..
```
Build with your preferred compiler. This was built and tested in Visual Studio 2019 x64.

### Timing example
By default, this builds with the timing example. This example tests the timing of the authors' SQPnP implementation against this one. In order to build this, recursively clone the repository to pull Eigen code as a submodule. To build without this, either add the cmake flag `-DBUILD_EXAMPLES=OFF` or `-DBUILD_TIMING_EXAMPLE=OFF` to disable examples or the timing example respectively.

You should build this example with the cmake flag `-DCMAKE_BUILD_TYPE=Release` in order to get a more accurate timing estimate.

## Expected input types
This code was written with integration into OpenCV's solvePnP method in mind, so it expects that the imagePoints are already normalized and undistorted. You can do this using `cv::undistortPoints`.