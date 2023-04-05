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

set(CUDA_HEADERS Source/SimulationEngine/SimulationEngineModes/del/SimulationEngineCuda.cuh Source/Stuff/cuda/cuda_image_creator.cuh Source/Stuff/cuda/get_device_count.cuh)
set(CUDA_SOURCES Source/SimulationEngine/SimulationEngineModes/del/SimulationEngineCuda.cu Source/Stuff/cuda/cuda_image_creator.cu Source/Stuff/cuda/get_device_count.cu)

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
    include_directories(Source/Stuff/external/moviemaker/include)
    include_directories(${GNUTLS_INCLUDR_DIRS})
    include_directories(${GTK3_INCLUDE_DIRS})
    include_directories(${RSVG_INCLUDE_DIR})
    include_directories(${SWSCALE_INCLUDE_DIRS})
    include_directories(${X264_INCLUDE_DIR})
#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lswresample -lm -lz")

    set(ADDITIONAL_PROJECT_FILES Source/Stuff/external/moviemaker/include/movie.h Source/Stuff/external/moviemaker/src/reader.cpp Source/Stuff/external/moviemaker/src/writer.cpp Source/UIWindows/Recorder/RecorderWindowUI.h Source/UIWindows/Recorder/Recorder.cpp Source/UIWindows/Recorder/Recorder.h Source/UIWindows/Recorder/RecorderSlots.cpp)
endif()

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CUDA_FILES ${CUDA_SOURCES} ${CUDA_HEADERS} Source/WorldRecorder/RecordingReconstructorCUDA.cu Source/WorldRecorder/RecordingReconstructorCUDA.cuh)
endif()

set(PROJECT_FILES Source/SimulationEngine/SimulationEngine.cpp Source/SimulationEngine/SimulationEngine.h Source/UIWindows/MainWindow/MainWindow.cpp Source/UIWindows/MainWindow/MainWindow.h Source/Organism/Organism.cpp Source/Organism/Organism.h Source/GridStuff/BaseGridBlock.h Source/Containers/ColorContainer.h Source/Organism/Anatomy/Anatomy.cpp Source/Organism/Anatomy/Anatomy.h Source/Stuff/enums/BlockTypes.hpp Source/Containers/SimulationParameters.h Source/Containers/EngineControlParametersContainer.h Source/Containers/OrganismBlockParameters.h Source/Stuff/Linspace.h Source/UIWindows/MainWindow/WindowUI.h Source/UIWindows/MainWindow/MainWindowSlots.cpp Source/UIWindows/MainWindow/MainWindowEvents.cpp Source/Organism/Brain/Brain.cpp Source/Organism/Brain/Brain.h Source/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThread.cpp Source/Stuff/structs/Actions.h Source/PRNGS/xorshf96.h Source/Stuff/ImageStuff/textures.h Source/Organism/Brain/Observation.h Source/GridStuff/STGridWorld.h Source/UIWindows/MainWindow/MainWindowDataSavingStuff.cpp Source/UIWindows/Statistics/Statistics.h Source/UIWindows/OrganismEditor/OrganismEditor.h Source/UIWindows/OrganismEditor/OrganismEditor.cpp Source/UIWindows/OrganismEditor/OrganismEditorSlots.cpp Source/Stuff/UIMisc.h Source/Stuff/UIMisc.cpp Source/Stuff/external/PerlinNoise.hpp Source/UIWindows/OrganismEditor/OrganismEditorDataSaving.cpp Source/UIWindows/InfoWindow/InfoWindowUI.h Source/UIWindows/InfoWindow/InfoWindow.cpp Source/UIWindows/InfoWindow/InfoWindow.h Source/Stuff/ImageStuff/ImageCreation.h Source/Stuff/ImageStuff/ImageCreation.cpp Source/Stuff/DataSavingFunctions.h Source/Stuff/DataSavingFunctions.cpp Source/UIWindows/WorldEvents/WorldEvents.cpp Source/UIWindows/WorldEvents/WorldEvents.h Source/UIWindows/WorldEvents/EventNodes.h Source/UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidget.cpp Source/UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidget.h Source/UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidget.cpp Source/UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidget.h Source/UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidgetSlots.cpp Source/UIWindows/WorldEvents/Misc/ParametersList.h Source/UIWindows/WorldEvents/Misc/ParametersList.cpp Source/UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidgetSlots.cpp Source/Containers/OrganismInfoContainer.h Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsController.cpp Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsSlots.cpp Source/UIWindows/Benchmark/Benchmarks.cpp Source/UIWindows/Benchmark/Benchmarks.h Source/UIWindows/Benchmark/BenchmarkSlots.cpp Source/SimulationEngine/OrganismsController.cpp Source/SimulationEngine/OrganismsController.h Source/Organism/OCC/OrganismConstructionCode.cpp Source/Organism/OCC/OrganismConstructionCode.h Source/Organism/Anatomy/SimpleAnatomyMutationLogic.h Source/Containers/OrganismConstructionCodeParameters.h Source/Containers/OCCLogicContainer.h Source/UIWindows/OCCParameters/OCCParameters.cpp Source/UIWindows/OCCParameters/OCCParameters.h Source/UIWindows/OCCParameters/OCCParametersSlots.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.h Source/WorldRecorder/WorldRecorder.cpp Source/WorldRecorder/WorldRecorder.h Source/WorldRecorder/RecordingReconstructor.cpp Source/WorldRecorder/RecordingReconstructor.h Source/Stuff/structs/ConstMap.h Source/Organism/Anatomy/AnatomyContainers.h Source/UIWindows/Statistics/Statistics.cpp Source/Organism/Anatomy/AnatomyCountersMap.h Source/Stuff/external/ArrayView.h Source/Stuff/CoordinateCalculation/iterate_between_two_points.h Source/UIWindows/MainWindow/MainWindowImageCreation.cpp Source/Stuff/CoordinateCalculation/get_max_coordinate.h Source/Stuff/CoordinateCalculation/AngleRotationConversion.h Source/Stuff/CoordinateCalculation/AngleConversion.h)

add_executable(${PROJECT_NAME} main.cpp ${PROJECT_FILES} ${CUDA_FILES} ${ADDITIONAL_PROJECT_FILES})

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

if (TESTING)
    find_package(Catch2 3 REQUIRED)

    include(CTest)
    include(Catch)

    set(TESTS Tests/Organism/Organism.cpp)

    foreach(item ${TESTS})
        #       regex is a devil's toy
        #       extracting file name without fluff
        string(REGEX MATCH "([^\\/]*$)" name_item "${item}")
        string(REPLACE ".cpp" "" name_item "${name_item}")

        add_executable(${name_item} ${item} ${PROJECT_FILES} ${CUDA_FILES} ${ADDITIONAL_PROJECT_FILES} Source/Stuff/CoordinateCalculation/AngleRotationConversion.h)
        target_link_libraries(${name_item} Catch2::Catch2WithMain)
        catch_discover_tests(${name_item})

        if (CUDAToolkit_FOUND)
            set_target_properties(${name_item} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        endif()
        target_link_libraries(${name_item}
            Qt6::Core
            Qt6::Gui
            Qt6::Widgets
#           Qt6::Charts
            )
        if (Boost_FOUND)
            target_link_libraries(${name_item} ${Boost_LIBRARIES})
        endif()

        if (NOT NO_RECORDER)
            if (WIN_COMPILATION)
                target_link_libraries(${name_item} ${FFMPEG_LIBRARIES} ${GTK3_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY} ZLIB::ZLIB)
            else()
                target_link_libraries(${name_item} ${FFMPEG_LIBRARIES} ${GTK3_LIBRARIES} ${X264_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY} ZLIB::ZLIB)
            endif()
        endif()
    endforeach()
endif()