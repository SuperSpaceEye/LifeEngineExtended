set(MIN_VERSION 3.22)
cmake_minimum_required(VERSION ${MIN_VERSION})
set(PROJECT_NAME LifeEngineExtended)
project(${PROJECT_NAME})

option(NO_RECORDER "If you don't want recorder" ON)

#option(BOOST_ROOT "If on windows, set path to boost root directory" "C:\\libs\\boost\\boost_1_79_0")
#option(BOOST_LIBRARYDIR "If on windows, set path to compiled libraries, like 'boost\\stage\\libs'" "C:\\libs\\boost\\boost_1_79_0\\stage\\libs")

set(NO_RECORDER ON)

set(TESTING 0)

#set(DEBUG 1)
#set(INLINE 1)
#set(SANITIZE 1)
set(DEBUG 0)
set(INLINE 0)
set(SANITIZE 0)

# -ffast-math -funroll-loops -ftree-parallelize-loops=4

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -w")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_PREFIX_PATH "/home/spaceeye/QtSource/6.4.0/SrcWASM/qtbase")
set(CMAKE_FIND_ROOT_PATH "/home/spaceeye/QtSource/6.4.0/SrcWASM/qtbase")
set(CMAKE_TOOLCHAIN_FILE "/home/spaceeye/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")
set(CMAKE_CROSSCOMPILING_EMULATOR "/home/spaceeye/emsdk/node/14.18.2_64bit/bin/node")
set(-DQT_DEBUG_FIND_PACKAGE=ON)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -03")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart=shared")

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


#include_directories(/usr/include/GLES2)

#set(GLESv2_INCLUDE_DIR "/usr/include")
#find_path(INCDIR NAMES "GLES2/gl2.h")
#message(STATUS ${INCDIR})
set(GLESv2_LIBRARY "/usr/lib/x86_64-linux-gnu/libGLESv2.so")
set(EGL_INCLUDE_DIR "/usr/include")
#message(STATUS ${GLESv2_LIBRARY})
#set(HAVE_GLESv2 ON)

#set(GLESv2_INCLUDE_DIR /home/spaceeye/Qt/6.4.0/wasm_32/mkspecs/common/uikit/GLES2/)
#set(GLESv2_LIBRARY "/home/spaceeye/Qt/6.4.0/wasm_32/mkspecs/common/uikit/GLES2/gl2.h")

find_package(Qt6 COMPONENTS
        Core
        Gui
        Widgets
#        Charts
        REQUIRED)


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

add_compile_definitions(__EMSCRIPTEN_COMPILATION__)
include_directories("Source")

set(PROJECT_FILES Source/SimulationEngine/SimulationEngine.cpp Source/SimulationEngine/SimulationEngine.h Source/UIWindows/MainWindow/MainWindow.cpp Source/UIWindows/MainWindow/MainWindow.h Source/Organism/Organism.cpp Source/Organism/Organism.h Source/GridStuff/BaseGridBlock.h Source/Containers/ColorContainer.h Source/Organism/Anatomy/Anatomy.cpp Source/Organism/Anatomy/Anatomy.h Source/Stuff/enums/BlockTypes.hpp Source/Containers/SimulationParameters.h Source/Containers/EngineControlParametersContainer.h Source/Containers/OrganismBlockParameters.h Source/Stuff/Linspace.h Source/UIWindows/MainWindow/WindowUI.h Source/UIWindows/MainWindow/MainWindowSlots.cpp Source/UIWindows/MainWindow/MainWindowEvents.cpp Source/Organism/Brain/Brain.cpp Source/Organism/Brain/Brain.h Source/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThread.cpp Source/Stuff/structs/Actions.h Source/PRNGS/xorshf96.h Source/Stuff/ImageStuff/textures.h Source/Organism/Brain/Observation.h Source/GridStuff/STGridWorld.h Source/UIWindows/MainWindow/MainWindowDataSavingStuff.cpp Source/UIWindows/Statistics/Statistics.h Source/UIWindows/OrganismEditor/OrganismEditor.h Source/UIWindows/OrganismEditor/OrganismEditor.cpp Source/UIWindows/OrganismEditor/OrganismEditorSlots.cpp Source/Stuff/UIMisc.h Source/Stuff/UIMisc.cpp Source/Stuff/external/PerlinNoise.hpp Source/UIWindows/OrganismEditor/OrganismEditorDataSaving.cpp Source/UIWindows/InfoWindow/InfoWindowUI.h Source/UIWindows/InfoWindow/InfoWindow.cpp Source/UIWindows/InfoWindow/InfoWindow.h Source/Stuff/ImageStuff/ImageCreation.h Source/Stuff/ImageStuff/ImageCreation.cpp Source/Stuff/DataSavingFunctions.h Source/Stuff/DataSavingFunctions.cpp Source/UIWindows/WorldEvents/WorldEvents.cpp Source/UIWindows/WorldEvents/WorldEvents.h Source/UIWindows/WorldEvents/EventNodes.h Source/UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidget.cpp Source/UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidget.h Source/UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidget.cpp Source/UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidget.h Source/UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidgetSlots.cpp Source/UIWindows/WorldEvents/Misc/ParametersList.h Source/UIWindows/WorldEvents/Misc/ParametersList.cpp Source/UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidgetSlots.cpp Source/Containers/OrganismInfoContainer.h Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsController.cpp Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsSlots.cpp Source/UIWindows/Benchmark/Benchmarks.cpp Source/UIWindows/Benchmark/Benchmarks.h Source/UIWindows/Benchmark/BenchmarkSlots.cpp Source/SimulationEngine/OrganismsController.cpp Source/SimulationEngine/OrganismsController.h Source/Organism/OCC/OrganismConstructionCode.cpp Source/Organism/OCC/OrganismConstructionCode.h Source/Organism/Anatomy/SimpleAnatomyMutationLogic.h Source/Containers/OrganismConstructionCodeParameters.h Source/Containers/OCCLogicContainer.h Source/UIWindows/OCCParameters/OCCParameters.cpp Source/UIWindows/OCCParameters/OCCParameters.h Source/UIWindows/OCCParameters/OCCParametersSlots.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.h Source/WorldRecorder/WorldRecorder.cpp Source/WorldRecorder/WorldRecorder.h Source/WorldRecorder/RecordingReconstructor.cpp Source/WorldRecorder/RecordingReconstructor.h Source/Stuff/structs/ConstMap.h Source/Organism/Anatomy/AnatomyContainers.h Source/UIWindows/Statistics/Statistics.cpp Source/Organism/Anatomy/AnatomyCountersMap.h Source/Stuff/external/ArrayView.h Source/Stuff/iterate_between_two_points.h Source/UIWindows/MainWindow/MainWindowImageCreation.cpp)

qt_add_executable(${PROJECT_NAME} main.cpp ${PROJECT_FILES} ${CUDA_FILES} ${ADDITIONAL_PROJECT_FILES})

message(STATUS "Setting compilation target to WASM")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-s WASM=1 -s USE_PTHREADS -s ALLOW_MEMORY_GROWTH -sPTHREAD_POOL_SIZE=10 -s ASYNCIFY -s EXPORTED_RUNTIME_METHODS='[\"cwrap\", \"stringToUTF16\"]'")

if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDAToolkit version ${CUDAToolkit_VERSION}")

    add_compile_definitions(__CUDA_USED__)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "No CUDAToolkit found.")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
#            Qt6::Charts
)


if (NOT NO_RECORDER)
    target_link_libraries(${PROJECT_NAME} ${GTK3_LIBRARIES} ${X264_LIBRARIES} ${FFMPEG_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY})
endif()