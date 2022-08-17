//
// Created by spaceeye on 16.08.22.
//

#include "SimulationEngineSingleThreadBenchmark.h"

SimulationEngineSingleThreadBenchmark::SimulationEngineSingleThreadBenchmark() {

}

using SC = SerializedOrganismBlockContainer;

void SimulationEngineSingleThreadBenchmark::create_benchmark_organisms() {
    auto brain = Brain(BrainTypes::SimpleBrain);
    auto anatomy = Anatomy();
    Organism * p_organism;

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2*i) {
            for (int y = -2*i; y <= 2*i; y += 2*i) {
                blocks.emplace_back(BlockTypes::ProducerBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        benchmark_organisms["produce_food"].emplace_back(p_organism);
    }
}

void SimulationEngineSingleThreadBenchmark::remove_benchmark_organisms() {
    for (auto & [key, value]: benchmark_organisms) {
        for (auto organism: value) {
            delete organism;
        }
        value.clear();
    }

    benchmark_organisms.clear();
}

void SimulationEngineSingleThreadBenchmark::init_benchmark() {
    create_benchmark_organisms();
}

void SimulationEngineSingleThreadBenchmark::finish_benchmarking() {
    remove_benchmark_organisms();
    benchmark_results.clear();
}

const std::vector<BenchmarkResult> & SimulationEngineSingleThreadBenchmark::get_results() {
    return &benchmark_results;
}