//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

void Recorder::le_number_of_pixels_per_block_slot() {
    le_slot_lower_bound<int>(num_pixels_per_block, num_pixels_per_block, "int", _ui.le_number_or_pixels_per_block, 1, "1");
}

void Recorder::b_create_image_slot() {

}