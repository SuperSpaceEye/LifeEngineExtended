cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME TheLifeEngineCpp)
#project(${PROJECT_NAME} LANGUAGES CUDA CXX)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)


# optimizations. They are not needed?
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")

#for perf
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline-functions")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline-functions")

#find_package(CUDA REQUIRED)
find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)

# either i am doing something wrong (more likely), or OpenMP only does worse.
#find_package(OpenMP)

add_executable(${PROJECT_NAME} main.cpp LifeEngineCpp/SimulationEngine.cpp LifeEngineCpp/SimulationEngine.h LifeEngineCpp/WindowCoreProgramLogic.cpp LifeEngineCpp/WindowCore.h LifeEngineCpp/Organism/Organism.cpp LifeEngineCpp/Organism/Organism.h LifeEngineCpp/Organism/Organism_parts/OrganismBlock.cpp LifeEngineCpp/Organism/Organism_parts/OrganismBlock.h LifeEngineCpp/GridBlocks/BaseGridBlock.cpp LifeEngineCpp/GridBlocks/BaseGridBlock.h LifeEngineCpp/ColorContainer.h LifeEngineCpp/Organism/Anatomy.cpp LifeEngineCpp/Organism/Anatomy.h LifeEngineCpp/BlockTypes.hpp LifeEngineCpp/SimulationParameters.h LifeEngineCpp/EngineControlContainer.h LifeEngineCpp/OrganismBlockParameters.h LifeEngineCpp/Linspace.h LifeEngineCpp/WindowUI.h LifeEngineCpp/WindowCoreSlots.cpp LifeEngineCpp/WindowCoreEvents.cpp LifeEngineCpp/Organism/Brain.cpp LifeEngineCpp/Organism/Brain.h LifeEngineCpp/SimulationEngineModes/SimulationEnginePartialMultiThread.cpp LifeEngineCpp/SimulationEngineModes/SimulationEnginePartialMultiThread.h LifeEngineCpp/SimulationEngineModes/SimulationEngineSingleThread.cpp)
#add_executable(${PROJECT_NAME} main.cpp)

if (CUDA_FOUND)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARATE_COMPILATION ON)
endif()

if (Qt5_FOUND)
    target_link_libraries(${PROJECT_NAME}
            Qt5::Core
            Qt5::Gui
            Qt5::Widgets
            )
endif()


#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()