//
// Created by spaceeye on 28.07.22.
//

#ifndef LIFEENGINEEXTENDED_RECORDINGCONTAINER_H
#define LIFEENGINEEXTENDED_RECORDINGCONTAINER_H

#include <vector>
#include <fstream>

#include "../../GridBlocks/BaseGridBlock.h"


enum class BufferFailConditions {
    AllIsWell,
    NotCompatibleWidth,
    NotCompatibleHeight,
    BufferPosBiggerThanBufferSize,
};

struct BufferFailure {
    BufferFailConditions result = BufferFailConditions::AllIsWell;
    int loaded_value = 0;
};

struct RecordingData {
    std::vector<std::vector<BaseGridBlock>> second_simulation_grid_buffer;
    std::string path_to_save;
    int buffer_size = 500;
    int buffer_pos = 0;
    int recorded_states = 0;
    int saved_buffers = 1;

    //For some reason during recording at some point the speed of saving drops significantly. Probably because I
    //write it on hard drive and not an SSD.
    static void save_buffer_to_disk(std::string & path_to_save, int buffer_pos,
                             int & saved_buffers, int width, int height,
                             std::vector<std::vector<BaseGridBlock>> & second_simulation_grid_buffer) {
        if (buffer_pos == 0) {
            return;
        }
        saved_buffers++;
        auto path = path_to_save + "/" + std::to_string(saved_buffers);
        std::ofstream out(path, std::ios::out | std::ios::binary);
        out.write((char*)&width, sizeof(int));
        out.write((char*)&height, sizeof(int));
        out.write((char*)&buffer_pos, sizeof(int));
        for (int i = 0; i < buffer_pos; i++) {
            auto & frame = second_simulation_grid_buffer[i];
            out.write((char*)&frame[0], sizeof(BaseGridBlock)*frame.size());
        }
        out.close();
    }

    static BufferFailure load_buffer_from_disk(std::string & buffer_path, int width, int height, int buffer_size, int & buffer_pos,
                                               std::vector<std::vector<BaseGridBlock>> & second_simulation_grid_buffer) {
        std::ifstream is(buffer_path, std::ios::in | std::ios::binary);
        int loaded_width;
        int loaded_height;
        int loaded_buffer_pos;

        is.read((char*)&loaded_width, sizeof(int));
        is.read((char*)&loaded_height, sizeof(int));
        is.read((char*)&loaded_buffer_pos, sizeof(int));

        if (loaded_width != width)           {return BufferFailure{BufferFailConditions::NotCompatibleWidth,            loaded_width}     ;}
        if (loaded_height != height)         {return BufferFailure{BufferFailConditions::NotCompatibleHeight,           loaded_height}    ;}
        if (loaded_buffer_pos > buffer_size) {return BufferFailure{BufferFailConditions::BufferPosBiggerThanBufferSize, loaded_buffer_pos};}

        buffer_pos = loaded_buffer_pos;

        for (int i = 0; i < loaded_buffer_pos; i++) {
            auto & frame = second_simulation_grid_buffer[i];
            is.read((char*)&frame[0], sizeof(BaseGridBlock)*frame.size());
        }

        is.close();
        return BufferFailure{};
    }

    static void load_info_buffer_data(std::string & buffer_path, int & width, int & height, int & buffer_pos) {
        std::ifstream is(buffer_path, std::ios::in | std::ios::binary);
        is.read((char*)&width, sizeof(int));
        is.read((char*)&height, sizeof(int));
        is.read((char*)&buffer_pos, sizeof(int));
        is.close();
    }
};

#endif //LIFEENGINEEXTENDED_RECORDINGCONTAINER_H
