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

    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::ProduceFood].emplace_back(
            OrganismContainer{"1 block", p_organism}
    );

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



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::EatFood].emplace_back(
            OrganismContainer{"1 mouth block", p_organism}
            );

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2) {
            for (int y = -2*i; y <= 2*i; y += 2) {
                blocks.emplace_back(BlockTypes::MouthBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        benchmark_organisms[BenchmarkTypes::EatFood].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " mouth blocks", p_organism}
        );
    }



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::KillerBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::ApplyDamage].emplace_back(
            OrganismContainer{"1 block", p_organism, -1}
    );



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    p_organism->damage = p_organism->calculate_max_life() + 1;
    benchmark_organisms[BenchmarkTypes::TickLifetime].emplace_back(
            OrganismContainer{"1 block", p_organism}
    );

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2) {
            for (int y = -2*i; y <= 2*i; y += 2) {
                blocks.emplace_back(BlockTypes::MouthBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        p_organism->damage = p_organism->calculate_max_life() + 1;
        benchmark_organisms[BenchmarkTypes::TickLifetime].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " blocks", p_organism}
        );
    }



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::EraseOrganisms].emplace_back(
            OrganismContainer{"1 block", p_organism}
    );

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2) {
            for (int y = -2*i; y <= 2*i; y += 2) {
                blocks.emplace_back(BlockTypes::MouthBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        benchmark_organisms[BenchmarkTypes::EraseOrganisms].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " blocks", p_organism}
        );
    }

    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::EyeBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::GetObservations].emplace_back(
            OrganismContainer{"1 block 1 space", p_organism}
    );
    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::EyeBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::GetObservations].emplace_back(
            OrganismContainer{"1 block 10 space", p_organism, 10}
    );
    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::EyeBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::GetObservations].emplace_back(
            OrganismContainer{"1 block 20 space", p_organism, 20}
    );
    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::EyeBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::GetObservations].emplace_back(
            OrganismContainer{"1 block 50 space", p_organism, 50}
    );



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::EyeBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::ThinkDecision].emplace_back(
            OrganismContainer{"none", p_organism}
    );



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MoverBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::RotateOrganism].emplace_back(
            OrganismContainer{"1 block", p_organism}
    );

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2) {
            for (int y = -2*i; y <= 2*i; y += 2) {
                blocks.emplace_back(BlockTypes::MoverBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        benchmark_organisms[BenchmarkTypes::RotateOrganism].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " blocks", p_organism}
        );
    }



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MoverBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::MoveOrganism].emplace_back(
            OrganismContainer{"1 block", p_organism}
    );

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -2*i; x <= 2*i; x += 2) {
            for (int y = -2*i; y <= 2*i; y += 2) {
                blocks.emplace_back(BlockTypes::MoverBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
        benchmark_organisms[BenchmarkTypes::MoveOrganism].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " blocks", p_organism}
        );
    }



    anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MoverBlock, Rotation::UP, 0, 0);
    p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);
    benchmark_organisms[BenchmarkTypes::TryMakeChild].emplace_back(
            OrganismContainer{"1 block", p_organism, 2}
    );

    for (int i = 1; i <= 3; i++) {
        anatomy = Anatomy();
        std::vector<SC> blocks;
        for (int x = -i; x <= i; x += 1) {
            for (int y = -i; y <= i; y += 1) {
                blocks.emplace_back(BlockTypes::MoverBlock, Rotation::UP, x, y);
            }
        }
        anatomy.set_many_blocks(blocks);
        p_organism = new Organism(0, 0, Rotation::UP, anatomy, brain, &sp, &bp, 1);

        auto distance = 2 * i + 2;
        benchmark_organisms[BenchmarkTypes::TryMakeChild].emplace_back(
                OrganismContainer{std::to_string(blocks.size()) + " blocks", p_organism, distance}
        );
    }
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

                place_organisms_of_type(benchmark_organism, num_organisms, res, benchmark_organism_data.additional_distance);
                gen.set_seed(seed);
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
                            prepare_eat_food_benchmark();
                            benchmark_eat_food(false, res);
                            break;
                        case BenchmarkTypes::ApplyDamage:
//                            prepare_apply_damage_benchmark();
                            benchmark_apply_damage(false, res);
                            break;
                        case BenchmarkTypes::TickLifetime:
                            prepare_tick_lifetime_benchmark();
                            benchmark_tick_lifetime(false, res);
                            break;
                        case BenchmarkTypes::EraseOrganisms:
                            prepare_erase_organisms_benchmark(benchmark_organism);
                            benchmark_erase_organisms(false, res);
                            break;
                        case BenchmarkTypes::ReserveOrganisms:
                            break;
                        case BenchmarkTypes::GetObservations:
                            benchmark_get_observations(false, res);
                            break;
                        case BenchmarkTypes::ThinkDecision:
                            benchmark_think_decision(false, res);
                            break;
                        case BenchmarkTypes::RotateOrganism:
                            benchmark_rotate_organism(false, res);
                            break;
                        case BenchmarkTypes::MoveOrganism:
                            benchmark_move_organism(false, res);
                            break;
                        case BenchmarkTypes::TryMakeChild:
                            prepare_try_make_child_benchmark();
                            benchmark_try_make_child(false, res);
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

void SimulationEngineSingleThreadBenchmark::place_organisms_of_type(Organism *organism, int num_organisms,
                                                                    BenchmarkResult &result, int additional_distance) {
//    auto dimensions = SimulationEngineSingleThread::get_organism_dimensions(organism);
//
//    bool continue_flag = false;
//
//    int x_step = std::abs(dimensions[0]) + dimensions[2] + 2 + additional_distance;
//    int y_step = std::abs(dimensions[1]) + dimensions[3] + 2 + additional_distance;
//
//    int x = 1;
//    int y = 1;
//
//    int i = 0;
//    for (; i < num_organisms; i++) {
//        x += x_step;
//        if (x >= dc.simulation_width)  { y += y_step; x = x_step; }
//        if (y >= dc.simulation_height) { break; }
//
//        organism->x = x;
//        organism->y = y;
//        for (auto & block: organism->anatomy._organism_blocks) {
//            if (SimulationEngineSingleThread::check_if_block_out_of_bounds(&dc, organism, block, organism->rotation)) { continue_flag = true;break;}
//
//            auto * w_block = &dc.CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
//            [organism->y + block.get_pos(organism->rotation).y];
//
//            if (w_block->type != BlockTypes::EmptyBlock)
//            { continue_flag = true;break;}
//        }
//        if (continue_flag) {continue_flag = false; i--; continue;}
//
//        auto new_organism = new Organism(organism);
//        dc.organisms.emplace_back(new_organism);
//        SimulationEngineSingleThread::place_organism(&dc, organism);
//    }
//    result.num_organisms = i;
}

void SimulationEngineSingleThreadBenchmark::reset_state() {
//    for (auto * organism: dc.organisms) {
//        delete organism;
//    }
//
//    dc.organisms.clear();
//    for (auto & row: dc.CPU_simulation_grid) {
//        row = std::vector<SingleThreadGridBlock>(dc.simulation_width);
//    }
//    for (int x = 0; x < dc.simulation_width; x++) {
//        dc.CPU_simulation_grid[x][0].type = BlockTypes::WallBlock;
//        dc.CPU_simulation_grid[x][dc.simulation_height - 1].type = BlockTypes::WallBlock;
//    }
//
//    for (int y = 0; y < dc.simulation_height; y++) {
//        dc.CPU_simulation_grid[0][y].type = BlockTypes::WallBlock;
//        dc.CPU_simulation_grid[dc.simulation_width - 1][y].type = BlockTypes::WallBlock;
//    }
//
//    gen.set_seed(seed);
}

#define NOW std::chrono::high_resolution_clock::now
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

void SimulationEngineSingleThreadBenchmark::prepare_produce_food_benchmark() {
    for (auto & row: dc.CPU_simulation_grid) {
        for (auto & block: row) {
            if (block.type == BlockTypes::FoodBlock) {block.type = BlockTypes::EmptyBlock;}
        }
    }
}
void SimulationEngineSingleThreadBenchmark::benchmark_produce_food(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//    for (auto * organism: organisms) {
//        start = NOW();
//        SimulationEngineSingleThread::produce_food(&dc, &sp, organism, gen);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_eat_food_benchmark() {
    for (auto & row: dc.CPU_simulation_grid) {
        for (auto & block: row) {
            if (block.type == BlockTypes::EmptyBlock) {block.type = BlockTypes::FoodBlock;}
        }
    }
}
void SimulationEngineSingleThreadBenchmark::benchmark_eat_food(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//    for (auto * organism: organisms) {
//        start = NOW();
//        SimulationEngineSingleThread::eat_food(&dc, &sp, organism);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_apply_damage_benchmark() {

}
void SimulationEngineSingleThreadBenchmark::benchmark_apply_damage(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//    for (auto * organism: organisms) {
//        start = NOW();
//        SimulationEngineSingleThread::apply_damage(&dc, &sp, organism);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_tick_lifetime_benchmark() {
//    for (auto * organism: dc.organisms) {
//        SimulationEngineSingleThread::place_organism(&dc, organism);
//    }
}
void
SimulationEngineSingleThreadBenchmark::benchmark_tick_lifetime(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//
//    std::vector<int> to_erase;
//    to_erase.reserve(organisms.size());
//
//    for (auto * organism: organisms) {
//        start = NOW();
//        SimulationEngineSingleThread::tick_lifetime(&dc, organism);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_erase_organisms_benchmark(Organism *organism) {
//    for (int i = 0; i < num_organisms; i++) {
//        dc.organisms.emplace_back(new Organism(organism));
//    }
}
void SimulationEngineSingleThreadBenchmark::benchmark_erase_organisms(bool randomized_organism_access,
                                                                      BenchmarkResult &res) {
//    std::vector<int> to_erase{};
//    to_erase.reserve(dc.organisms.size());
//    for (int i = 0; i < num_organisms; i++) {
//        to_erase.emplace_back(i);
//    }
//    auto start = NOW();
//    auto end  = NOW();
//
//    for (int i = 0; i < num_organisms; i++) {
//        start = NOW();
////        SimulationEngineSingleThread::erase_organisms(&dc, to_erase, i);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_reserve_organisms_benchmark() {}
void SimulationEngineSingleThreadBenchmark::benchmark_reserve_organisms() {}

void SimulationEngineSingleThreadBenchmark::prepare_get_observations_benchmark() {}
void
SimulationEngineSingleThreadBenchmark::benchmark_get_observations(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//
//    std::vector<std::vector<Observation>> observations;
//    observations.resize(dc.organisms.size(), std::vector<Observation>{Observation{}});
//
//    start = NOW();
////    SimulationEngineSingleThread::get_observations(&dc, &sp, organisms, observations);
//    end = NOW();
//    auto difference = duration_cast<nanoseconds>(end - start).count();
//    res.total_time_measured += difference;
//    res.num_tried += dc.organisms.size();
}

void SimulationEngineSingleThreadBenchmark::prepare_think_decision_benchmark() {}
void
SimulationEngineSingleThreadBenchmark::benchmark_think_decision(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//    std::vector<std::vector<Observation>> observations{dc.organisms.size()};
//    for (auto & observation: observations) {
//        observation = std::vector<Observation>{Observation{
//                static_cast<BlockTypes>(std::uniform_int_distribution<int>(0, 8)(gen)), 5, Rotation::UP
//                    }};
//    }
//
//    for (int i = 0; i < num_organisms; i++) {
//        auto * organism = organisms[i];
//        start = NOW();
//        organism->think_decision(observations[i], &gen);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_rotate_organism_benchmark() {}
void SimulationEngineSingleThreadBenchmark::benchmark_rotate_organism(bool randomized_organism_access,
                                                                      BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//
//    for (auto * organism: organisms) {
//        start = NOW();
//        SimulationEngineSingleThread::rotate_organism(&dc, organism, BrainDecision::RotateRight, &sp);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_move_organism_benchmark() {}
void
SimulationEngineSingleThreadBenchmark::benchmark_move_organism(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//
//    for (auto * organism: organisms) {
//        auto decision = static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 3)(gen));
//        start = NOW();
//        SimulationEngineSingleThread::move_organism(&dc, organism,
//                                                    decision,
//                                                    &sp);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}

void SimulationEngineSingleThreadBenchmark::prepare_try_make_child_benchmark() {
//    for (int i = num_organisms; i < dc.organisms.size(); i++) {
//        auto organism = dc.organisms[i];
//        for (auto & block: organism->anatomy._organism_blocks) {
//            auto * w_block = &dc.CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y];
//            w_block->type = BlockTypes::FoodBlock;
//            w_block->organism = nullptr;
//        }
//    }
//
//    dc.organisms.erase(dc.organisms.begin() + num_organisms, dc.organisms.end());
}

void
SimulationEngineSingleThreadBenchmark::benchmark_try_make_child(bool randomized_organism_access, BenchmarkResult &res) {
//    auto organisms = std::vector(dc.organisms);
//    if (randomized_organism_access) { std::shuffle(organisms.begin(), organisms.end(), gen); gen.set_seed(seed);}
//    auto start = NOW();
//    auto end  = NOW();
//
//    for (int i = 0; i < num_organisms; i++) {
//        auto organism = dc.organisms[i];
//        organism->food_collected = organism->food_needed+1;
//        start = NOW();
//        SimulationEngineSingleThread::try_make_child(&dc, &sp, organism, &gen);
//        end = NOW();
//        auto difference = duration_cast<nanoseconds>(end - start).count();
//        res.total_time_measured += difference;
//        res.num_tried++;
//    }
}