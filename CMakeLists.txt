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

set(CUDA_HEADERS Source/SimulationEngine/SimulationEngineModes/del/SimulationEngineCuda.cuh Source/Stuff/cuda_image_creator.cuh Source/Stuff/get_device_count.cuh)
set(CUDA_SOURCES Source/SimulationEngine/SimulationEngineModes/del/SimulationEngineCuda.cu Source/Stuff/cuda_image_creator.cu Source/Stuff/get_device_count.cu)

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
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Source/Stuff/moviemaker/include)
    include_directories(${GNUTLS_INCLUDR_DIRS})
    include_directories(${GTK3_INCLUDE_DIRS})
    include_directories(${RSVG_INCLUDE_DIR})
    include_directories(${SWSCALE_INCLUDE_DIRS})
    include_directories(${X264_INCLUDE_DIR})
#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lswresample -lm -lz")

    set(ADDITIONAL_PROJECT_FILES Source/Stuff/moviemaker/include/movie.h Source/Stuff/moviemaker/src/reader.cpp Source/Stuff/moviemaker/src/writer.cpp Source/UIWindows/Recorder/RecorderWindowUI.h Source/UIWindows/Recorder/Recorder.cpp Source/UIWindows/Recorder/Recorder.h Source/UIWindows/Recorder/RecorderSlots.cpp)
endif()

include_directories("Source/Stuff")

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CUDA_FILES ${CUDA_SOURCES} ${CUDA_HEADERS} Source/WorldRecorder/RecordingReconstructorCUDA.cu Source/WorldRecorder/RecordingReconstructorCUDA.cuh)
endif()

set(PROJECT_FILES main.cpp Source/SimulationEngine/SimulationEngine.cpp Source/SimulationEngine/SimulationEngine.h Source/UIWindows/MainWindow/MainWindow.cpp Source/UIWindows/MainWindow/MainWindow.h Source/Organism/CPU/Organism.cpp Source/Organism/CPU/Organism.h Source/GridStuff/BaseGridBlock.h Source/Containers/CPU/ColorContainer.h Source/Organism/CPU/Anatomy.cpp Source/Organism/CPU/Anatomy.h Source/Stuff/BlockTypes.hpp Source/Containers/CPU/SimulationParameters.h Source/Containers/CPU/EngineControlParametersContainer.h Source/Containers/CPU/OrganismBlockParameters.h Source/Stuff/Linspace.h Source/UIWindows/MainWindow/WindowUI.h Source/UIWindows/MainWindow/MainWindowSlots.cpp Source/UIWindows/MainWindow/MainWindowEvents.cpp Source/Organism/CPU/Brain.cpp Source/Organism/CPU/Brain.h Source/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThread.cpp Source/Stuff/Actions.h Source/PRNGS/xorshf96.h Source/Stuff/textures.h Source/Organism/CPU/ObservationStuff.h Source/GridStuff/STGridWorld.h Source/UIWindows/MainWindow/MainWindowDataSavingStuff.cpp Source/UIWindows/Statistics/Statistics.h Source/UIWindows/OrganismEditor/OrganismEditor.h Source/UIWindows/OrganismEditor/OrganismEditor.cpp Source/UIWindows/OrganismEditor/OrganismEditorSlots.cpp Source/Stuff/MiscFuncs.h Source/Stuff/MiscFuncs.cpp Source/Stuff/PerlinNoise.hpp Source/UIWindows/OrganismEditor/OrganismEditorDataSaving.cpp Source/UIWindows/InfoWindow/InfoWindowUI.h Source/UIWindows/InfoWindow/InfoWindow.cpp Source/UIWindows/InfoWindow/InfoWindow.h Source/Stuff/ImageCreation.h Source/Stuff/ImageCreation.cpp Source/Stuff/DataSavingFunctions.h Source/Stuff/DataSavingFunctions.cpp Source/UIWindows/WorldEvents/WorldEvents.cpp Source/UIWindows/WorldEvents/WorldEvents.h Source/UIWindows/WorldEvents/EventNodes.h Source/UIWindows/WorldEvents/ChangeValueEventNodeWidget.cpp Source/UIWindows/WorldEvents/ChangeValueEventNodeWidget.h Source/UIWindows/WorldEvents/ConditionalEventNodeWidget.cpp Source/UIWindows/WorldEvents/ConditionalEventNodeWidget.h Source/UIWindows/WorldEvents/ConditionalEventNodeWidgetSlots.cpp Source/UIWindows/WorldEvents/ParametersList.h Source/UIWindows/WorldEvents/ParametersList.cpp Source/UIWindows/WorldEvents/ChangeValueEventNodeWidgetSlots.cpp Source/Containers/CPU/OrganismInfoContainer.h Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsController.cpp Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsSlots.cpp Source/UIWindows/Benchmark/Benchmarks.cpp Source/UIWindows/Benchmark/Benchmarks.h Source/UIWindows/Benchmark/BenchmarkSlots.cpp Source/SimulationEngine/OrganismsController.cpp Source/SimulationEngine/OrganismsController.h Source/Organism/CPU/OrganismConstructionCode.cpp Source/Organism/CPU/OrganismConstructionCode.h Source/Organism/CPU/SimpleAnatomyMutationLogic.h Source/Containers/CPU/OrganismConstructionCodeParameters.h Source/Containers/CPU/OCCLogicContainer.h Source/UIWindows/OCCParameters/OCCParameters.cpp Source/UIWindows/OCCParameters/OCCParameters.h Source/UIWindows/OCCParameters/OCCParametersSlots.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.h Source/WorldRecorder/WorldRecorder.cpp Source/WorldRecorder/WorldRecorder.h Source/WorldRecorder/RecordingReconstructor.cpp Source/WorldRecorder/RecordingReconstructor.h Source/Stuff/ConstMap.h Source/Organism/CPU/AnatomyContainers.h Source/UIWindows/Statistics/Statistics.cpp Source/Organism/CPU/AnatomyCountersMap.h)

add_executable(${PROJECT_NAME} ${PROJECT_FILES} ${CUDA_FILES} ${ADDITIONAL_PROJECT_FILES})

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