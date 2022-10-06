set(MIN_VERSION 3.22)
cmake_minimum_required(VERSION ${MIN_VERSION})
set(PROJECT_NAME LifeEngineExtended)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

#https://developer.nvidia.com/cuda-gpus
#set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
set(CMAKE_CUDA_ARCHITECTURES 61)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -03")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart=shared")

#for perf
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline-functions")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline-functions")

#https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#set(CMAKE_LINK_DEF_FILE_FLAG "${CMAKE_LINK_DEF_FILE_FLAG} -fsanitize=address")

set(CUDA_HEADERS Source/SimulationEngine/SimulationEngineModes/SimulationEngineCuda.cuh Source/Stuff/cuda_image_creator.cuh Source/Stuff/get_device_count.cuh)
set(CUDA_SOURCES Source/SimulationEngine/SimulationEngineModes/SimulationEngineCuda.cu Source/Stuff/cuda_image_creator.cu Source/Stuff/get_device_count.cu)

set(EMSCRIPTEN_COMPILATION 1)

set(WIN_COMPILATION 0)
set(WIN_USE_CUDA_LIBS 0)

if(EMSCRIPTEN_COMPILATION)
    set(WIN_COMPILATION 0)
    set(WIN_USE_CUDA_LIBS 0)
endif()

if(NOT WIN_COMPILATION AND NOT EMSCRIPTEN_COMPILATION)
    set(WIN_USE_CUDA_LIBS 0)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
    set(CMAKE_NVCC_EXECUTABLE "/usr/local/cuda-11.7/bin")
endif()

if (NOT WIN_USE_CUDA_LIBS AND NOT WIN_COMPILATION AND NOT EMSCRIPTEN_COMPILATION)
    find_package(CUDAToolkit)
endif()

if (WIN_COMPILATION)
    set(BOOST_ROOT "C:\\libs\\boost\\boost_1_79_0")
    set(BOOST_LIBRARYDIR "C:\\libs\\boost\\boost_1_79_0\\stage\\libs")
    set(Boost_DEBUG 0)
    set(Boost_ARCHITECTURE -x64)
    set(Boost_LIB_PREFIX lib)

    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_DYNAMIC_LIBS ON)
    set(Boost_USE_RELEASE_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME ON)
endif()

find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        #        Charts
        REQUIRED)

if(NOT EMSCRIPTEN_COMPILATION)
    find_package(Qt5 COMPONENTS
            Core
            Gui
            Widgets
    #        Charts
            REQUIRED)

    find_package(Boost COMPONENTS random REQUIRED)

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    find_package(FFmpeg REQUIRED)
    if(NOT WIN_COMPILATION)
        find_package(Numa REQUIRED)
    else()
        set(NUMA_LIBRARY "")
    endif()
    find_package(PkgConfig REQUIRED)
    find_package(RSVG REQUIRED)
    find_package(Threads REQUIRED)
    find_package(x264 REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Source/Stuff/moviemaker/include)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    include_directories(${GTK3_INCLUDE_DIRS})
    include_directories(${RSVG_INCLUDE_DIR})
    include_directories(${SWSCALE_INCLUDE_DIRS})
    include_directories(${X264_INCLUDE_DIR})

    message(STATUS ${Boost_VERSION})

    set(ADDITIONAL_PROJECT_FILES Source/Stuff/moviemaker/include/movie.h Source/Stuff/moviemaker/src/reader.cpp Source/Stuff/moviemaker/src/writer.cpp Source/UIWindows/Recorder/RecorderWindowUI.h Source/UIWindows/Recorder/Recorder.cpp Source/UIWindows/Recorder/Recorder.h Source/UIWindows/Recorder/RecorderSlots.cpp)
#else()
endif()

add_compile_definitions(__VALGRIND_MODE__=0)
add_compile_definitions(__EMSCRIPTEN_COMPILATION__)

#find_package(OpenMP)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CUDA_FILES ${CUDA_SOURCES} ${CUDA_HEADERS})
#elseif(WIN_USE_CUDA_LIBS)
#    set(CUDA_FILES ${CUDA_HEADERS})
endif()

#add_library(CUDAFunctions SHARED IMPORTED)
#set_property(TARGET CUDAFunctions PROPERTY IMPORTED_LOCATION CUDACmake/Release/CUDAPartsDynamic.dll)
#set_property(TARGET CUDAFunctions PROPERTY IMPORTED_IMPLIB CUDACmake/Release/CUDAPartsDynamic.lib)

#Source/SimulationEngine/SimulationEngineModes/SimulationEnginePartialMultiThread.cpp Source/SimulationEngine/SimulationEngineModes/SimulationEnginePartialMultiThread.h

set(PROJECT_FILES main.cpp Source/SimulationEngine/SimulationEngine.cpp Source/SimulationEngine/SimulationEngine.h Source/UIWindows/MainWindow/MainWindow.cpp Source/UIWindows/MainWindow/MainWindow.h Source/Organism/CPU/Organism.cpp Source/Organism/CPU/Organism.h Source/GridBlocks/BaseGridBlock.h Source/Containers/CPU/ColorContainer.h Source/Organism/CPU/Anatomy.cpp Source/Organism/CPU/Anatomy.h Source/Stuff/BlockTypes.hpp Source/Containers/CPU/SimulationParameters.h Source/Containers/CPU/EngineControlParametersContainer.h Source/Containers/CPU/OrganismBlockParameters.h Source/Stuff/Linspace.h Source/UIWindows/MainWindow/WindowUI.h Source/UIWindows/MainWindow/MainWindowSlots.cpp Source/UIWindows/MainWindow/MainWindowEvents.cpp Source/Organism/CPU/Brain.cpp Source/Organism/CPU/Brain.h Source/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThread.cpp Source/Stuff/Actions.h Source/PRNGS/xorshf96.h Source/PRNGS/RandomGenerator.h Source/Stuff/textures.h Source/Organism/CPU/ObservationStuff.h Source/GridBlocks/SingleThreadGridBlock.h Source/UIWindows/MainWindow/MainWindowDataSavingStuff.cpp Source/UIWindows/Statistics/StatisticsCore.h Source/UIWindows/OrganismEditor/OrganismEditor.h Source/UIWindows/OrganismEditor/OrganismEditor.cpp Source/UIWindows/OrganismEditor/OrganismEditorSlots.cpp Source/Stuff/MiscFuncs.h Source/Stuff/MiscFuncs.cpp Source/Stuff/PerlinNoise.hpp Source/UIWindows/OrganismEditor/OrganismEditorDataSaving.cpp Source/UIWindows/InfoWindow/InfoWindowUI.h Source/UIWindows/InfoWindow/InfoWindow.cpp Source/UIWindows/InfoWindow/InfoWindow.h Source/Containers/CPU/RecordingContainer.h Source/Stuff/ImageCreation.h Source/Stuff/ImageCreation.cpp Source/Stuff/DataSavingFunctions.h Source/Stuff/DataSavingFunctions.cpp Source/UIWindows/WorldEvents/WorldEvents.cpp Source/UIWindows/WorldEvents/WorldEvents.h Source/UIWindows/WorldEvents/EventNodes.h Source/UIWindows/WorldEvents/ChangeValueEventNodeWidget.cpp Source/UIWindows/WorldEvents/ChangeValueEventNodeWidget.h Source/UIWindows/WorldEvents/ConditionalEventNodeWidget.cpp Source/UIWindows/WorldEvents/ConditionalEventNodeWidget.h Source/UIWindows/WorldEvents/ConditionalEventNodeWidgetSlots.cpp Source/UIWindows/WorldEvents/ParametersList.h Source/UIWindows/WorldEvents/ParametersList.cpp Source/UIWindows/WorldEvents/ChangeValueEventNodeWidgetSlots.cpp Source/Containers/CPU/OrganismInfoContainer.h Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsController.cpp Source/UIWindows/WorldEvents/WorldEventsController.h Source/UIWindows/WorldEvents/WorldEventsSlots.cpp Source/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThreadBenchmark.cpp Source/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThreadBenchmark.h Source/UIWindows/Benchmark/Benchmarks.cpp Source/UIWindows/Benchmark/Benchmarks.h Source/UIWindows/Benchmark/BenchmarkSlots.cpp Source/SimulationEngine/OrganismsController.cpp Source/SimulationEngine/OrganismsController.h Source/Organism/CPU/OrganismConstructionCode.cpp Source/Organism/CPU/OrganismConstructionCode.h Source/Organism/CPU/LegacyAnatomyMutationLogic.h Source/Organism/CPU/LegacyAnatomyMutationLogic.cpp Source/Containers/CPU/OrganismConstructionCodeParameters.h Source/Containers/CPU/OCCLogicContainer.h Source/UIWindows/OCCParameters/OCCParameters.cpp Source/UIWindows/OCCParameters/OCCParameters.h Source/UIWindows/OCCParameters/OCCParametersSlots.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.cpp Source/UIWindows/OrganismEditor/OCCTranspiler/OCCTranspiler.h Source/WorldRecorder/WorldRecorder.cpp Source/WorldRecorder/WorldRecorder.h Source/WorldRecorder/RecordingReconstructor.cpp Source/WorldRecorder/RecordingReconstructor.h)

add_executable(${PROJECT_NAME} ${PROJECT_FILES} ${CUDA_FILES} ${ADDITIONAL_PROJECT_FILES})
#target_link_libraries(${PROJECT_NAME} CUDAFunctions)

#if (WIN_USE_CUDA_LIBS)
#    foreach(ITEM IN LISTS CUDA_SOURCES)
#        set(CUDA_LIST_SOURCES "${CUDA_LIST_SOURCES}${ITEM} ")
#    endforeach()
#
#    add_custom_target(
#            run_CUDA_cmake ALL
#            COMMAND cmake CUDACmake/CmakeLists.txt -DCUDA_SOURCES="${CUDA_SOURCES}" -DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS}" -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" -DCMAKE_CUDA_STANDARD="${CMAKE_CUDA_STANDARD}" -DMIN_VERSION="${MIN_VERSION}" -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE
#            DEPENDS ${CUDA_SOURCES}
#            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#            #            USES_TERMINAL
#            COMMENT "Building dynamic CUDA lib."
#    )
#    add_dependencies(${PROJECT_NAME} run_CUDA_cmake)
#    add_custom_target(
#            run_CUDA_build ALL
#            COMMAND cmake --build . --config Release
#            DEPENDS ${CUDA_SOURCES}
#            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/CUDACmake
#            #            USES_TERMINAL
#            COMMENT "Building dynamic CUDA lib."
#    )
#    add_dependencies(${PROJECT_NAME} run_CUDA_build)
#endif ()

if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDAToolkit version ${CUDAToolkit_VERSION}")

    add_compile_definitions(__CUDA_USED__=1)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#    target_link_libraries(${PROJECT_NAME} CUDA::cudart)
elseif(WIN_USE_CUDA_LIBS)
    add_compile_definitions(__CUDA_USED__=1)
else()
    message(STATUS "No CUDAToolkit found.")
    add_compile_definitions(__CUDA_USED__=0)
endif()

if (Qt5_FOUND)
    if(NOT WIN_COMPILATION)
        #https://gitlab.kitware.com/cmake/cmake/-/issues/16915
        get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
        string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
        set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
        set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
        set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
    endif()

    target_link_libraries(${PROJECT_NAME}
            Qt5::Core
            Qt5::Gui
            Qt5::Widgets
#            Qt5::Charts
            )
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

if(NOT EMSCRIPTEN_COMPILATION)
    if (WIN_COMPILATION)
        target_link_libraries(${PROJECT_NAME} ${GTK3_LIBRARIES} ${FFMPEG_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY})
    else()
        target_link_libraries(${PROJECT_NAME} ${GTK3_LIBRARIES} ${X264_LIBRARIES} ${FFMPEG_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARY} ${NUMA_LIBRARY})
    endif()
endif()