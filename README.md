# ImprovedAssociationTrack-cpp

C++ implementation of Improved Association Pipeline that does not include an object detection algorithm.

## Overview

- The implementation is based on [An Improved Assocation Pipeline](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) 
and [Vertical-Beach](https://github.com/Vertical-Beach)'s implementation of [ByteTrack-cpp](https://github.com/Vertical-Beach/ByteTrack-cpp)
- Only tracking algorithm are implemented in this repository
  - Any object detection algorithm can be easily combined
- Provided as a shared library usable in C++17 or higher
- The output of the implementation has *NOT YET* been verified.
- Will be verified to MOT20.

## Dependencies

- Eigen 3.3
- C++ compiler with C++17 or higher support
- CMake 3.14 or higher
- Opencv 4.5 or higher
- TensorRT 11 or higher
- GoogleTest 1.10 or higher (Only tests)

## Build and Test

The shared library (libbytetrack.so) can be build with following commands:

```shell
mkdir build && cd build
cmake ..
make
```

The implementation can be test with following commands:

```shell
mkdir build && cd build
cmake .. -DBUILD_BYTETRACK_TEST=ON
make
ctest --verbose
```

## Tips

You can use docker container to build and test the implementation.

```shell
docker build . -t bytetrack-cpp:latest
docker run -ti --rm \
           -v ${PWD}:/usr/src/app \
           -w /usr/src/app \
           bytetrack-cpp:latest
```

## License

MIT
