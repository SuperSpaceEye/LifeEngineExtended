set(MIN_VERSION 3.22)
cmake_minimum_required(VERSION ${MIN_VERSION})
set(PROJECT_NAME LifeEngineExtended)
project(${PROJECT_NAME})

option(WIN_COMPILATION "If compiling for windows" OFF)
option(NO_RECORDER "If you don't want recorder" ON)
#set(NO_RECORDER ON)
#option(Qt6_DIR "Set path to qt6 binary" "/home/spaceeye/qt6/qt6-build/lib/cmake/Qt6")
option(USE_MARCH_NATIVE "Use -march=native option" OFF)

#set(Qt6_DIR "/home/spaceeye/qt6/qt6-build/qtbase/lib/cmake/Qt6")
#set(CMAKE_PREFIX_PATH "/home/spaceeye/qt6/qt6-build/qtbase/lib/cmake")

option(BOOST_ROOT "If on windows, set path to boost root directory" "C:\\libs\\boost\\boost_1_79_0")
option(BOOST_LIBRARYDIR "If on windows, set path to compiled libraries, like 'boost\\stage\\libs'" "C:\\libs\\boost\\boost_1_79_0\\stage\\libs")

set(NO_RECORDER ON)

set(TESTING 1)

#set(DEBUG 1)
#set(INLINE 1)
#set(SANITIZE 1)
set(DEBUG 0)
set(INLINE 0)
set(SANITIZE 0)

# -ffast-math -funroll-loops -ftree-parallelize-loops=4

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

include_directories("Source")

if(USE_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

#https://developer.nvidia.com/cuda-gpus
#set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
set(CMAKE_CUDA_ARCHITECTURES 61)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -03")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart=shared")

if(DEBUG)
    add_compile_definitions(__DEBUG__)
endif ()
if (INLINE OR DEBUG)
    #for perf
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline-functions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline-functions")
endif ()
if(SANITIZE OR DEBUG)
    #https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_LINK_DEF_FILE_FLAG "${CMAKE_LINK_DEF_FILE_FLAG} -fsanitize=address")
endif()

set(CUDA_HEADERS )
set(CUDA_SOURCES )

if(NOT WIN_COMPILATION)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
    set(CMAKE_NVCC_EXECUTABLE "/usr/local/cuda-11.8/bin")
endif()

if (NOT WIN_COMPILATION)
    find_package(CUDAToolkit)
endif()

if (WIN_COMPILATION)
    set(Boost_DEBUG 0)
    set(Boost_ARCHITECTURE -x64)
    set(Boost_LIB_PREFIX lib)

    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_DYNAMIC_LIBS ON)
    set(Boost_USE_RELEASE_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME ON)
endif()

find_package(Qt6 COMPONENTS
        Core
        Gui
        Widgets
#        Charts
        REQUIRED)

find_package(Boost COMPONENTS random REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (NO_RECORDER)
    add_compile_definitions(__NO_RECORDER__)
endif()

if (NOT NO_RECORDER)
    find_package(PkgConfig REQUIRED)
    if(NOT WIN_COMPILATION)
        find_package(NUMA REQUIRED)
    else()
        set(NUMA_LIBRARY "")
    endif()
    find_package(FFmpeg REQUIRED)
    find_package(GnuTLS REQUIRED)
    find_package(RSVG REQUIRED)
    find_package(Threads REQUIRED)
    find_package(x264 REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    find_package(ZLIB REQUIRED)

    include_directories(${FFMPEG_INCLUDE_DIRS})
    include_directories(Source/common/external/moviemaker/include)
    include_directories(${GNUTLS_INCLUDR_DIRS})
    include_directories(${GTK3_INCLUDE_DIRS})
    include_directories(${RSVG_INCLUDE_DIR})
    include_directories(${SWSCALE_INCLUDE_DIRS})
    include_directories(${X264_INCLUDE_DIR})
#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lswresample -lm -lz")

    set(ADDITIONAL_PROJECT_FILES )
endif()

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CUDA_FILES ${CUDA_SOURCES} ${CUDA_HEADERS})
endif()

add_compile_definitions(__FRONTEND_IMPL_RELATIVE_PATH__="qt/all_frontend_includes.h")

set(FRONTEND_IMPL_FILES Source/Frontend/qt/all_frontend_includes.h Source/Frontend/qt/impl/QtWindow.h)

set(PROJECT_FILES Source/Frontend/frontend_interface.h Source/Frontend/base_frontend.h Source/common/common_concepts.h Source/common/includes/qt.h Source/common/external/PerlinNoise.hpp Source/common/external/ArrayView.h Source/common/external/PRNGS/lehmer64.h Source/common/external/PRNGS/splitmix64.h Source/MainWindow/MainWindow.cpp Source/MainWindow/MainWindow.h)
add_executable(${PROJECT_NAME} main.cpp ${PROJECT_FILES} ${FRONTEND_IMPL_FILES} ${CUDA_FILES} ${ADDITIONAL_PROJECT_FILES})

if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDAToolkit version ${CUDAToolkit_VERSION}")

    add_compile_definitions(__CUDA_USED__)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "No CUDAToolkit found.")
endif()

if(CUDAToolkit_FOUND)
    #should be set for each qt6 component for some reason
    #https://gitlab.kitware.com/cmake/cmake/-/issues/16915
    get_property( core_options TARGET Qt6::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt6::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt6::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )

    get_property( core_options TARGET Qt6::Gui PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt6::Gui PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt6::Gui PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )

    get_property( core_options TARGET Qt6::Widgets PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt6::Widgets PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt6::Widgets PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )

    set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()

target_link_libraries(${PROJECT_NAME}
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
#            Qt6::Charts
        )

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

if (NOT NO_RECORDER)
    if (WIN_COMPILATION)
        target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBRARIES} ${GTK3_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY} ZLIB::ZLIB)
    else()
        target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBRARIES} ${GTK3_LIBRARIES} ${X264_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY} ZLIB::ZLIB)
    endif()
endif()