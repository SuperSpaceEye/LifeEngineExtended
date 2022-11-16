//
// Created by spaceeye on 03.11.22.
//

#ifndef LIFEENGINEEXTENDED_IPSSMOOTHER_H
#define LIFEENGINEEXTENDED_IPSSMOOTHER_H

#include <vector>

struct IPSSmoother {
private:
    int max_items = 0;

    int front_cursor = 0;
    int end_cursor = 0;

    double frames = 0;
    std::vector<double> data;
public:
    IPSSmoother()=default;
    void set_max_items(int max_items) {
        this->max_items = max_items;
        frames = 0;
        data = std::vector<double>{};
        data.resize(max_items+1, 0);
        front_cursor = 0;
        end_cursor = 1;
    }
    //How tf do you make it right???
    // https://stackoverflow.com/questions/5614018/am-i-calculating-my-fps-correctly
    // https://stackoverflow.com/questions/59165392/c-how-to-make-precise-frame-rate-limit
    // first n log will be inaccurate but i don't care
    void log_data(double frames) {
        data[front_cursor] = frames;
        this->frames += frames;

        front_cursor++;
        this->frames -= data[end_cursor];
        end_cursor++;

        if (front_cursor >= max_items+1) {front_cursor = 0;}
        if (end_cursor >= max_items+1) {end_cursor = 0;}
    }
    double get_rate_per_second() {
        return frames / (data.size()-1);
    }
};

#endif //LIFEENGINEEXTENDED_IPSSMOOTHER_H