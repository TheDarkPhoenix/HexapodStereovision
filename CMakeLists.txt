cmake_minimum_required(VERSION 2.8)
project( kam )

set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_INCLUDE_DIRS
#	/usr/local/include
#	/usr/local/include/opencv2
#)

#set(OpenCV_LIB_DIR
#	/usr/local/lib
#)

set(OpenCV_INCLUDE_DIRS
	/home/maciek/opencv/opencv-master/include
)

set(OpenCV_LIB_DIR
	/home/maciek/opencv/build5/lib
)

set(OpenCV_LIBS
	opencv_world
	opencv_contrib_world
	pthread
)

include_directories( ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
link_directories( ${OpenCV_LIB_DIR} )
add_executable( kam main.cpp)
target_link_libraries( kam ${OpenCV_LIBS} )
