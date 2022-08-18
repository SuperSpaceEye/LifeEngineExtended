//
// Created by spaceeye on 16.08.22.
//

#include "SimulationEngineSingleThreadBenchmark.h"

SimulationEngineSingleThreadBenchmark::SimulationEngineSingleThreadBenchmark() {
    resize_benchmark_grid(1000, 1000);
    reset_state();
    init_benchmark();
}

bool SimulationEngineSingleThreadBenchmark::resize_benchmark_grid(int width, int height) {
    if (benchmark_running) {return false;}
    dc.simulation_width  = width;
    dc.simulation_height = height;
    dc.CPU_simulation_grid.resize(dc.simulation_width, std::vector<SingleThreadGridBlock>(dc.simulation_height, SingleThreadGridBlock{}));
    return true;
}

using SC = SerializedOrganismBlockContainer;

void SimulationEngineSingleThreadBenchmark::create_benchmark_organisms() {
    auto brain = Brain(BrainTypes::SimpleBrain);
    auto anatomy = Anatomy();
    Organism * p_organism;

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2) {
            for (int y = -2*i; y <= 2*i; y += 2) {
                blocks.emplace_back(BlockTypes::ProducerBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        benchmark_organisms[BenchmarkTypes::ProduceFood].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " producing blocks", p_organism}
                );
    }
    auto & vector = benchmark_organisms[BenchmarkTypes::ProduceFood];
}

void SimulationEngineSingleThreadBenchmark::remove_benchmark_organisms() {
    for (auto & [key, value]: benchmark_organisms) {
        for (const auto& data: value) {
            delete data.organism;
        }
        value.clear();
    }

    benchmark_organisms.clear();
}

void SimulationEngineSingleThreadBenchmark::init_benchmark() {
    if (initialized) { return;}
    create_benchmark_organisms();
    initialized = true;
}

void SimulationEngineSingleThreadBenchmark::finish_benchmarking() {
    stop_benchmark = true;
    while (benchmark_running) {}
    stop_benchmark = false;
    initialized = false;
    remove_benchmark_organisms();
    benchmark_results.clear();
}

const std::vector<BenchmarkResult> & SimulationEngineSingleThreadBenchmark::get_results() {
    return benchmark_results;
}

void SimulationEngineSingleThreadBenchmark::start_benchmarking(const std::vector<BenchmarkTypes>& benchmarks_to_do) {
    if (benchmark_running) { return;}
    benchmark_running = true;
    std::thread thread([&, benchmarks_to_do]() {
        for (auto & benchmark: benchmarks_to_do) {
            for (auto & benchmark_organism_data: benchmark_organisms[benchmark]) {
                auto * benchmark_organism = benchmark_organism_data.organism;
                benchmark_results.emplace_back(BenchmarkResult{num_organisms, 0, 0, 0, benchmark, benchmark_organism_data.additional_data});
                auto &res = benchmark_results.back();

                place_organisms_of_type(benchmark_organism, num_organisms, res);
                for (int i = 0; i < num_iterations; i++) {
                    if (stop_benchmark) {
                        reset_state();
                        stop_benchmark = false;
                        benchmark_running = false;
                        return;
                    }
                    res.num_iterations++;
                    switch (benchmark) {
                        case BenchmarkTypes::ProduceFood:
                            prepare_produce_food_benchmark();
                            benchmark_produce_food(false, res);
                            break;
                        case BenchmarkTypes::EatFood:
                            break;
                        case BenchmarkTypes::ApplyDamage:
                            break;
                        case BenchmarkTypes::TickLifetime:
                            break;
                        case BenchmarkTypes::EraseOrganisms:
                            break;
                        case BenchmarkTypes::ReserveOrganisms:
                            break;
                        case BenchmarkTypes::GetObservations:
                            break;
                        case BenchmarkTypes::ThinkDecision:
                            break;
                        case BenchmarkTypes::RotateOrganism:
                            break;
                        case BenchmarkTypes::MoveOrganism:
                            break;
                        case BenchmarkTypes::TryMakeChild:
                            break;
                    }
                }
                reset_state();
            }
        }
        benchmark_running = false;
    });
    thread.detach();
}

void SimulationEngineSingleThreadBenchmark::place_organisms_of_type(Organism *organism, int num_organisms, BenchmarkResult &result) {
    auto dimensions = SimulationEngineSingleThread::get_organism_dimensions(organism);

    bool continue_flag = false;

    int x_step = std::abs(dimensions[0]) + dimensions[2] + 2;
    int y_step = std::abs(dimensions[1]) + dimensions[3] + 2;

    int x = 1;
    int y = 1;

    int i = 0;
    for (; i < num_organisms; i++) {
        x += x_step;
        if (x >= dc.simulation_width)  { y += y_step; x = x_step; }
        if (y >= dc.simulation_height) { break; }

        organism->x = x;
        organism->y = y;
        for (auto & block: organism->anatomy._organism_blocks) {
            if (SimulationEngineSingleThread::check_if_block_out_of_bounds(&dc, organism, block, organism->rotation)) { continue_flag = true;break;}

            auto * w_block = &dc.CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
            [organism->y + block.get_pos(organism->rotation).y];

            if (w_block->type != BlockTypes::EmptyBlock)
            { continue_flag = true;break;}
        }
        if (continue_flag) {continue_flag = false; i--; continue;}

        auto new_organism = new Organism(organism);
        dc.organisms.emplace_back(new_organism);
        SimulationEngineSingleThread::place_organism(&dc, organism);
    }
    result.num_organisms = i;
}

void SimulationEngineSingleThreadBenchmark::reset_state() {
    for (auto * organism: dc.organisms) {
        delete organism;
    }

    dc.organisms.clear();
    for (auto & row: dc.CPU_simulation_grid) {
        row = std::vector<SingleThreadGridBlock>(dc.simulation_width);
    }
    for (int x = 0; x < dc.simulation_width; x++) {
        dc.CPU_simulation_grid[x][0].type = BlockTypes::WallBlock;
        dc.CPU_simulation_grid[x][dc.simulation_height - 1].type = BlockTypes::WallBlock;
    }

    for (int y = 0; y < dc.simulation_height; y++) {
        dc.CPU_simulation_grid[0][y].type = BlockTypes::WallBlock;
        dc.CPU_simulation_grid[dc.simulation_width - 1][y].type = BlockTypes::WallBlock;
    }

    gen.set_seed(seed);
}

void SimulationEngineSingleThreadBenchmark::prepare_produce_food_benchmark() {
    for (auto & row: dc.CPU_simulation_grid) {
        for (auto & block: row) {
            if (block.type == BlockTypes::FoodBlock) {block.type = BlockTypes::EmptyBlock;}
        }
    }
}

#define NOW std::chrono::high_resolution_clock::now
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

void
SimulationEngineSingleThreadBenchmark::benchmark_produce_food(bool randomized_organism_access, BenchmarkResult &res) {
    auto organisms = std::vector(dc.organisms);
    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
    auto start = NOW();
    auto end  = NOW();
    for (auto * organism: organisms) {
        start = NOW();
        SimulationEngineSingleThread::produce_food(&dc, &sp, organism, gen);
        end = NOW();
        auto difference = duration_cast<nanoseconds>(end - start).count();
        res.total_time_measured += difference;
        res.num_tried++;
    }
}