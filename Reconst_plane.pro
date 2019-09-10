QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        Source/sfm_train.cpp \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

CONFIG      *= link_pkgconfig
PKGCONFIG   *= opencv4
PKGCONFIG   *= eigen3
PKGCONFIG   *= metslib
PKGCONFIG   *= zcm

INCLUDEPATH += /usr/local/include/vtk-8.2 \
               /usr/local/include/pcl-1.9 \
               ../zcm_types/camera_basler/cpp_types \
               ../zcm_types/service/cpp_types

# for PCL
#PKGCONFIG   += pcl_2d-1.9
#PKGCONFIG   += pcl_common-1.9
#PKGCONFIG   += pcl_features-1.9
#PKGCONFIG   += pcl_filters-1.9
#PKGCONFIG   += pcl_geometry-1.9
#PKGCONFIG   += pcl_io-1.9
#PKGCONFIG   += pcl_kdtree-1.9
#PKGCONFIG   += pcl_keypoints-1.9
#PKGCONFIG   += pcl_ml-1.9
#PKGCONFIG   += pcl_octree-1.9
#PKGCONFIG   += pcl_outofcore-1.9
#PKGCONFIG   += pcl_people-1.9
#PKGCONFIG   += pcl_recognition-1.9
#PKGCONFIG   += pcl_registration-1.9
#PKGCONFIG   += pcl_sample_consensus-1.9
#PKGCONFIG   += pcl_search-1.9
#PKGCONFIG   += pcl_segmentation-1.9
#PKGCONFIG   += pcl_stereo-1.9
#PKGCONFIG   += pcl_surface-1.9
#PKGCONFIG   += pcl_tracking-1.9
#PKGCONFIG   += pcl_visualization-1.9

INCLUDEPATH += /usr/local/include/vtk-8.0

LIBS += -lboost_system \
        -lvtkCommonCore-8.2 \
        -lvtkRenderingCore-8.2 \
        -lvtkCommonDataModel-8.2  \
        -lvtkCommonMath-8.2 \
        -lvtkFiltersSources-8.2 \
        -lvtkCommonExecutionModel-8.2 \
        -lvtkRenderingLOD-8.2 

HEADERS += \
    Header/sfm_train.h
