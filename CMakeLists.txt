cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME TheLifeEngineCpp)
#project(${PROJECT_NAME} LANGUAGES CUDA CXX)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -pthread)

# optimizations. They are not needed?
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#find_package(CUDA REQUIRED)
find_package(SFML 2 REQUIRED COMPONENTS audio graphics window system network)
# for some fucking reason tgui-0.10 doesn't want to install and i am too noob at linux and cmake to resolve the fucking reason why
find_package(TGUI 0.9)

# either i am doing something wrong (more likely), or OpenMP only does worse.
#find_package(OpenMP)

add_executable(${PROJECT_NAME} main.cpp LifeEngineCpp/SimulationEngine.cpp LifeEngineCpp/SimulationEngine.h LifeEngineCpp/UIWindow.cpp LifeEngineCpp/UIWindow.h LifeEngineCpp/Organisms/Organism.cpp LifeEngineCpp/Organisms/Organism.h LifeEngineCpp/Organisms/Organism_parts/OrganismBlock.cpp LifeEngineCpp/Organisms/Organism_parts/OrganismBlock.h LifeEngineCpp/GridBlocks/BaseGridBlock.cpp LifeEngineCpp/GridBlocks/BaseGridBlock.h LifeEngineCpp/ColorContainer.h LifeEngineCpp/Organisms/Anatomy.cpp LifeEngineCpp/Organisms/Anatomy.h LifeEngineCpp/BlockTypes.h LifeEngineCpp/UIElemets/GridLayout2D.cpp LifeEngineCpp/UIElemets/GridLayout2D.h LifeEngineCpp/ParametersStruct.h LifeEngineCpp/SimulationModes.h LifeEngineCpp/EngineControlContainer.h LifeEngineCpp/OrganismBlockParameters.h LifeEngineCpp/Linspace.h)
#add_executable(${PROJECT_NAME} main.cpp)

if (CUDA_FOUND)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARATE_COMPILATION ON)
endif()

if (SFML_FOUND)
    target_link_libraries(${PROJECT_NAME} sfml-graphics sfml-system sfml-window sfml-audio sfml-network)
endif()

if (TGUI_FOUND)
    target_link_libraries(${PROJECT_NAME} tgui)
endif()

#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()