set(MIN_VERSION 3.22)
cmake_minimum_required(VERSION ${MIN_VERSION})
set(PROJECT_NAME TheLifeEngineCpp)
project(${PROJECT_NAME})

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
set(CMAKE_NVCC_EXECUTABLE "/usr/local/cuda-11.7/bin")
#https://developer.nvidia.com/cuda-gpus
#set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(AUTO_UI_FILE_REBUILDING 0)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart=shared")

#for perf
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline-functions")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline-functions")

set(CUDA_HEADERS LifeEngineCpp/SimulationEngine/SimulationEngineModes/SimulationEngineCuda.cuh LifeEngineCpp/Organism/CUDA/CUDA_Anatomy.cuh LifeEngineCpp/Organism/CUDA/CUDA_Brain.cuh LifeEngineCpp/Organism/CUDA/CUDA_Organism.cuh LifeEngineCpp/Containers/CUDA/CUDAEngineDataContainer.cuh LifeEngineCpp/Stuff/cuda_image_creator.cuh LifeEngineCpp/Stuff/get_device_count.cuh)
set(CUDA_SOURCES LifeEngineCpp/SimulationEngine/SimulationEngineModes/SimulationEngineCuda.cu LifeEngineCpp/Organism/CUDA/CUDA_Anatomy.cu LifeEngineCpp/Organism/CUDA/CUDA_Brain.cu LifeEngineCpp/Organism/CUDA/CUDA_Organism.cu LifeEngineCpp/Stuff/cuda_image_creator.cu LifeEngineCpp/Stuff/get_device_count.cu)

set(WIN_COMPILATION 0)

set(WIN_USE_CUDA_LIBS 0)

if(NOT WIN_COMPILATION)
    set(WIN_USE_CUDA_LIBS 0)
endif()

if (NOT WIN_USE_CUDA_LIBS AND NOT WIN_COMPILATION)
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
        REQUIRED)


find_package(Boost COMPONENTS random REQUIRED)

message(STATUS ${Boost_VERSION})

add_compile_definitions(__VALGRIND_MODE__=0)

#find_package(OpenMP)

if(CUDAToolkit_FOUND)
    set(CUDA_FILES ${CUDA_SOURCES} ${CUDA_HEADERS})
#elseif(WIN_USE_CUDA_LIBS)
#    set(CUDA_FILES ${CUDA_HEADERS})
endif()

#add_library(CUDAFunctions SHARED IMPORTED)
#set_property(TARGET CUDAFunctions PROPERTY IMPORTED_LOCATION CUDACmake/Release/CUDAPartsDynamic.dll)
#set_property(TARGET CUDAFunctions PROPERTY IMPORTED_IMPLIB CUDACmake/Release/CUDAPartsDynamic.lib)

set(PROJECT_FILES main.cpp LifeEngineCpp/SimulationEngine/SimulationEngine.cpp LifeEngineCpp/SimulationEngine/SimulationEngine.h LifeEngineCpp/MainWindow/WindowCoreProgramLogic.cpp LifeEngineCpp/MainWindow/WindowCore.h LifeEngineCpp/Organism/CPU/Organism.cpp LifeEngineCpp/Organism/CPU/Organism.h LifeEngineCpp/GridBlocks/BaseGridBlock.h LifeEngineCpp/Containers/CPU/ColorContainer.h LifeEngineCpp/Organism/CPU/Anatomy.cpp LifeEngineCpp/Organism/CPU/Anatomy.h LifeEngineCpp/Stuff/BlockTypes.hpp LifeEngineCpp/Containers/CPU/SimulationParameters.h LifeEngineCpp/Containers/CPU/EngineControlContainer.h LifeEngineCpp/Containers/CPU/OrganismBlockParameters.h LifeEngineCpp/Stuff/Linspace.h LifeEngineCpp/MainWindow/WindowUI.h LifeEngineCpp/MainWindow/WindowCoreSlots.cpp LifeEngineCpp/MainWindow/WindowCoreEvents.cpp LifeEngineCpp/Organism/CPU/Brain.cpp LifeEngineCpp/Organism/CPU/Brain.h LifeEngineCpp/SimulationEngine/SimulationEngineModes/SimulationEnginePartialMultiThread.cpp LifeEngineCpp/SimulationEngine/SimulationEngineModes/SimulationEnginePartialMultiThread.h LifeEngineCpp/SimulationEngine/SimulationEngineModes/SimulationEngineSingleThread.cpp LifeEngineCpp/Stuff/Actions.h LifeEngineCpp/PRNGS/xorshf96.h LifeEngineCpp/PRNGS/RandomGenerator.h LifeEngineCpp/Stuff/pix_pos.h LifeEngineCpp/Stuff/textures.h LifeEngineCpp/Organism/CPU/ObservationStuff.h LifeEngineCpp/GridBlocks/AtomicGridBlock.h LifeEngineCpp/MainWindow/WindowCoreDataSavingStuff.cpp LifeEngineCpp/CustomJsonParser/json_parser.hpp LifeEngineCpp/CustomJsonParser/write.hpp LifeEngineCpp/Statistics/StatisticsCore.h LifeEngineCpp/OrganismEditor/OrganismEditor.h LifeEngineCpp/OrganismEditor/OrganismEditor.cpp LifeEngineCpp/OrganismEditor/OrganismEditorSlots.cpp LifeEngineCpp/Stuff/MiscFuncs.h LifeEngineCpp/Stuff/MiscFuncs.cpp LifeEngineCpp/Stuff/PerlinNoise.hpp)

add_executable(${PROJECT_NAME} ${PROJECT_FILES} ${CUDA_FILES})
#target_link_libraries(${PROJECT_NAME} CUDAFunctions)

if (AUTO_UI_FILE_REBUILDING)
    add_custom_target(
            run_window ALL
            COMMAND uic -g cpp "${CMAKE_SOURCE_DIR}/UiFiles/MainWindow/mainwindow.ui" -o "${CMAKE_SOURCE_DIR}/LifeEngineCpp/MainWindow/WindowUI.h"
            BYPRODUCTS WindowUI.h
            COMMENT "Generating Main Window header file."
    )

    add_custom_target(
            run_statistics ALL
            COMMAND uic -g cpp "${CMAKE_SOURCE_DIR}/UiFiles/Statistics/statistics.ui" -o "${CMAKE_SOURCE_DIR}/LifeEngineCpp/Statistics/StatisticsUI.h"
            BYPRODUCTS StatisticsUI.h
            COMMENT "Generating Statistics header file."
    )

    add_custom_target(
            run_editor ALL
            COMMAND uic -g cpp "${CMAKE_SOURCE_DIR}/UiFiles/OrganismEditor/editor.ui" -o "${CMAKE_SOURCE_DIR}/LifeEngineCpp/OrganismEditor/EditorUI.h"
            BYPRODUCTS EditorUI.h
            COMMENT "Generating Organism editor's header file."
    )

    add_dependencies(${PROJECT_NAME} run_window)
    add_dependencies(${PROJECT_NAME} run_statistics)
    add_dependencies(${PROJECT_NAME} run_editor)

endif()

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
            )
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()