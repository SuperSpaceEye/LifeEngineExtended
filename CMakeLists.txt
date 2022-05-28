cmake_minimum_required(VERSION 3.20)
set(PROJECT_NAME TheLifeEngineCpp)
project(${PROJECT_NAME})

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
set(CMAKE_NVCC_EXECUTABLE "/usr/local/cuda-11.7/bin")
#https://developer.nvidia.com/cuda-gpus
#set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
set(CMAKE_CUDA_ARCHITECTURES 62)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

#for perf
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline-functions")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline-functions")

find_package(CUDA)
find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)

find_package(Boost COMPONENTS random REQUIRED)

#find_package(OpenMP)
if(CUDA_FOUND)
    add_compile_definitions(__CUDA_USED__=1)
    set(CUDA_SOURCES LifeEngineCpp/SimulationEngineModes/SimulationEngineCuda.cuh LifeEngineCpp/SimulationEngineModes/SimulationEngineCuda.cu LifeEngineCpp/GridBlocks/AtomicGridBlock.h LifeEngineCpp/Organism/CUDA/CUDA_Anatomy.cpp LifeEngineCpp/Organism/CUDA/CUDA_Anatomy.h LifeEngineCpp/Organism/CUDA/CUDA_Brain.cpp LifeEngineCpp/Organism/CUDA/CUDA_Brain.h LifeEngineCpp/Organism/CUDA/CUDA_Organism.cpp LifeEngineCpp/Organism/CUDA/CUDA_Organism.h LifeEngineCpp/Containers/CUDA/CUDAEngineDataContainer.h)
else()
    add_compile_definitions(__CUDA_USED__=0)
endif()

add_executable(${PROJECT_NAME} main.cpp LifeEngineCpp/SimulationEngine.cpp LifeEngineCpp/SimulationEngine.h LifeEngineCpp/WindowCoreProgramLogic.cpp LifeEngineCpp/WindowCore.h LifeEngineCpp/Organism/CPU/Organism.cpp LifeEngineCpp/Organism/CPU/Organism.h LifeEngineCpp/GridBlocks/BaseGridBlock.cpp LifeEngineCpp/GridBlocks/BaseGridBlock.h LifeEngineCpp/Containers/CPU/ColorContainer.h LifeEngineCpp/Organism/CPU/Anatomy.cpp LifeEngineCpp/Organism/CPU/Anatomy.h LifeEngineCpp/BlockTypes.hpp LifeEngineCpp/Containers/CPU/SimulationParameters.h LifeEngineCpp/Containers/CPU/EngineControlContainer.h LifeEngineCpp/Containers/CPU/OrganismBlockParameters.h LifeEngineCpp/Linspace.h LifeEngineCpp/WindowUI.h LifeEngineCpp/WindowCoreSlots.cpp LifeEngineCpp/WindowCoreEvents.cpp LifeEngineCpp/Organism/CPU/Brain.cpp LifeEngineCpp/Organism/CPU/Brain.h LifeEngineCpp/SimulationEngineModes/SimulationEnginePartialMultiThread.cpp LifeEngineCpp/SimulationEngineModes/SimulationEnginePartialMultiThread.h LifeEngineCpp/SimulationEngineModes/SimulationEngineSingleThread.cpp LifeEngineCpp/OrganismEditor.cpp LifeEngineCpp/OrganismEditor.h LifeEngineCpp/Actions.h ${CUDA_SOURCES})

if (Qt5_FOUND)
    #https://gitlab.kitware.com/cmake/cmake/-/issues/16915
    get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
    set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )

    target_link_libraries(${PROJECT_NAME}
            Qt5::Core
            Qt5::Gui
            Qt5::Widgets
            )
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

if (CUDA_FOUND)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()