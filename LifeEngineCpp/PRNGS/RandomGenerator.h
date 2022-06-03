//
// Created by spaceeye on 03.06.22.
//

#ifndef THELIFEENGINECPP_RANDOMGENERATOR_H
#define THELIFEENGINECPP_RANDOMGENERATOR_H

#include <cstdint>
#include <boost/nondet_random.hpp>
#include <boost/random.hpp>

#include "lehmer64.h"
#include "splitmix64.h"
#include "xorshf96.h"
#include "shufflebag.h"


enum class RandomGeneratorType {
    //Boost types
    rand48,
    kreutzer1986,
    taus88,
    mt11213b,
    mt19937,
    mt19937_64,
    laggedFibonacci23209,
    laggedFibonacci44497,

    //Custom types
    lehmer64,
    xorshf96,
    splitmix64,
    shufflebag,
};

class RandomGenerator {
private:
    boost::random_device rd;

    bool shufflebag_initialized = false;

    RandomGeneratorType type;
public:
    boost::rand48 rand48;
    boost::kreutzer1986 kreutzer1986;
    boost::taus88 taus88;
    boost::mt11213b mt11213b;
    boost::mt19937 mt19937;
    boost::mt19937_64 mt19937_64;
    boost::lagged_fibonacci23209 laggedFibonacci23209;
    boost::lagged_fibonacci44497 laggedFibonacci44497;


    lehmer64 lehmer;
    xorshf96 xorshf;
    splitmix64 splitmix;
    shufflebag shufflebag_;

    RandomGenerator()=default;
    explicit RandomGenerator(RandomGeneratorType type) {
        rand48 = boost::rand48(rd());
        kreutzer1986 = boost::kreutzer1986(rd());
        taus88 = boost::taus88(rd());
        mt11213b = boost::mt11213b(rd());
        mt19937 = boost::mt19937(rd());
        mt19937_64 = boost::mt19937_64(rd());
        laggedFibonacci23209 = boost::lagged_fibonacci23209(rd());
        laggedFibonacci44497 = boost::lagged_fibonacci44497(rd());


        lehmer = lehmer64(rd());
        xorshf = xorshf96(rd(), rd(), rd());
        splitmix = splitmix64(rd());

        this->type = type;
    }

    template<class Generator>
    void init_shufflebag(Generator &gen, int sequence_length) {shufflebag_.template initialize(gen, sequence_length); shufflebag_initialized = true;}
    void init_shufflebag(std::vector<uint64_t> &states) {shufflebag_.initialize(states); shufflebag_initialized = true;}

    using result_type = uint64_t;
    constexpr result_type min() {
        switch (type) {
            case RandomGeneratorType::rand48:
                return boost::rand48::min();
            case RandomGeneratorType::kreutzer1986:
                return boost::kreutzer1986::min();
            case RandomGeneratorType::taus88:
                return boost::taus88::min();
            case RandomGeneratorType::mt11213b:
                return boost::mt11213b::min();
            case RandomGeneratorType::mt19937:
                return boost::mt19937::min();
            case RandomGeneratorType::mt19937_64:
                return boost::mt19937_64::min();
            case RandomGeneratorType::laggedFibonacci23209:
                return boost::lagged_fibonacci23209::min();
            case RandomGeneratorType::laggedFibonacci44497:
                return boost::lagged_fibonacci44497::min();
            case RandomGeneratorType::lehmer64:
                return lehmer64::min();
            case RandomGeneratorType::xorshf96:
                return xorshf96::min();
            case RandomGeneratorType::splitmix64:
                return splitmix64::min();
            case RandomGeneratorType::shufflebag:
                if (shufflebag_initialized) { return shufflebag_.min();}
                else {throw "Shuffle bag is not initialized";}
        }
    }
    constexpr result_type max() {
        switch (type) {
            case RandomGeneratorType::rand48:
                return boost::rand48::max();
            case RandomGeneratorType::kreutzer1986:
                return boost::kreutzer1986::max();
            case RandomGeneratorType::taus88:
                return boost::taus88::max();
            case RandomGeneratorType::mt11213b:
                return boost::mt11213b::max();
            case RandomGeneratorType::mt19937:
                return boost::mt19937::max();
            case RandomGeneratorType::mt19937_64:
                return boost::mt19937_64::max();
            case RandomGeneratorType::laggedFibonacci23209:
                return boost::lagged_fibonacci23209::max();
            case RandomGeneratorType::laggedFibonacci44497:
                return boost::lagged_fibonacci44497::max();
            case RandomGeneratorType::lehmer64:
                return lehmer64::max();
            case RandomGeneratorType::xorshf96:
                return xorshf96::max();
            case RandomGeneratorType::splitmix64:
                return splitmix64::max();
            case RandomGeneratorType::shufflebag:
                if (shufflebag_initialized) { return shufflebag_.max();}
                else {throw "Shuffle bag is not initialized";}
        }
    }
    //The generation time is like 10 times bigger if you get numbers through this method for some reason.
    result_type operator()() {
        switch (type) {
            case RandomGeneratorType::rand48:
                return rand48();
            case RandomGeneratorType::kreutzer1986:
                return kreutzer1986();
            case RandomGeneratorType::taus88:
                return taus88();
            case RandomGeneratorType::mt11213b:
                return mt11213b();
            case RandomGeneratorType::mt19937:
                return mt19937();
            case RandomGeneratorType::mt19937_64:
                return mt19937_64();
            case RandomGeneratorType::laggedFibonacci23209:
                return laggedFibonacci23209();
            case RandomGeneratorType::laggedFibonacci44497:
                return laggedFibonacci44497();
            case RandomGeneratorType::lehmer64:
                return lehmer();
            case RandomGeneratorType::xorshf96:
                return xorshf();
            case RandomGeneratorType::splitmix64:
                return splitmix();
            case RandomGeneratorType::shufflebag:
                if (shufflebag_initialized) { return shufflebag_();}
                else {throw "Shuffle bag is not initialized";}
        }
    }

    void change_type(RandomGeneratorType new_type) {
        if (new_type == RandomGeneratorType::shufflebag && !shufflebag_initialized) {throw "Shuffle bag is not initialized";}
        this->type = type;
    }
    RandomGeneratorType get_type() {return type;}
};

#endif //THELIFEENGINECPP_RANDOMGENERATOR_H


//#include <boost/nondet_random.hpp>
//#include <boost/random.hpp>
//#include <random>
//#include <chrono>
//#include "LifeEngineCpp/PRNGS/lehmer64.h"
//#include "LifeEngineCpp/PRNGS/xorshf96.h"
//#include "LifeEngineCpp/PRNGS/splitmix64.h"
//#include "LifeEngineCpp/PRNGS/shufflebag.h"
//
//#include "LifeEngineCpp/PRNGS/RandomGenerator.h"
//int num_iter = 10'000'000;
//auto rd = std::random_device();
//auto lehmer = lehmer64(rd());
//
//RandomGenerator gen(RandomGeneratorType::lehmer64);
//gen.init_shufflebag(gen.laggedFibonacci44497, num_iter);
//
//auto point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.mt19937);
//}
//auto end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" mt19937" << std::endl;
//
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.taus88);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" taus88" << std::endl;
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.rand48);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" rand48" << std::endl;
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.mt11213b);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" mt11213b" << std::endl;
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.laggedFibonacci44497);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" lagged_fibonacci44497" << std::endl;
//
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" lehmer64" << std::endl;
//
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(lehmer);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" lehmer64" << std::endl;
//
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.xorshf);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" xorshf96" << std::endl;
//
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.splitmix);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" splitmix64" << std::endl;
//
//point = std::chrono::high_resolution_clock::now();
//for (int i = 0; i < num_iter; i++) {
//volatile auto num = std::uniform_int_distribution<int>(0, 9)(gen.shufflebag_);
//}
//end_point = std::chrono::high_resolution_clock::now();
//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - point).count() <<
//" shuffle bag" << std::endl;
