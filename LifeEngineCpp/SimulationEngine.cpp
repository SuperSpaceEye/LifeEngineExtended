//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"
#include "Organisms/Rotation.h"

SimulationEngine::SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                                   OrganismBlockParameters& organism_block_parameters, SimulationParameters& simulation_parameters,
                                   std::mutex& mutex):
    mutex(mutex), dc(engine_data_container), cp(engine_control_parameters), op(organism_block_parameters), sp(simulation_parameters){

    mt = std::mt19937{rd()};
    dist = std::uniform_int_distribution<int>{0, 8};
}

//TODO refactor pausing/pass_tick/synchronise_tick
void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();
    while (cp.engine_working) {
//        if (cp.calculate_simulation_tick_delta_time) {point = std::chrono::high_resolution_clock::now();}
        //it works better without mutex... huh.
        //std::lock_guard<std::mutex> guard(mutex);
        if (cp.stop_engine) {
            kill_threads();
            cp.engine_working = false;
            cp.engine_paused = true;
            cp.stop_engine = false;
            return;
        }
        if (cp.change_simulation_mode) { change_mode(); }
        if (cp.build_threads) { build_threads(); }
        if (cp.engine_pause || cp.engine_global_pause) { cp.engine_paused = true; } else {cp.engine_paused = false;}
        process_user_action_pool();
        if (!cp.engine_paused || cp.engine_pass_tick) {
            //if ((!cp.engine_pause || cp.synchronise_simulation_tick) && !cp.engine_global_pause) {
                cp.engine_paused = false;
                cp.engine_pass_tick = false;
                cp.synchronise_simulation_tick = false;
                simulation_tick();
//                if (cp.calculate_simulation_tick_delta_time) {dc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
//                if (!dc.unlimited_simulation_fps) {std::this_thread::sleep_for(std::chrono::microseconds(int(dc.simulation_interval * 1000000 - dc.delta_time)));}
            //}
        }
    }
}

void SimulationEngine::change_mode() {
    if (cp.change_to_mode == cp.simulation_mode) {
        return;
    }

    switch (cp.change_to_mode) {
        case SimulationModes::CPU_Single_Threaded:
            if (cp.simulation_mode == SimulationModes::CPU_Multi_Threaded) {
                kill_threads();
            }
            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {

            }
            break;
        case SimulationModes::CPU_Multi_Threaded:
            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {

            }
            build_threads();
            break;
        case SimulationModes::GPU_CUDA_mode:
            if (cp.simulation_mode == SimulationModes::CPU_Multi_Threaded) {
                kill_threads();
            }
            break;
    }
    cp.simulation_mode = cp.change_to_mode;
}

void SimulationEngine::simulation_tick() {
    dc.engine_ticks++;

    switch (cp.simulation_mode){
        case SimulationModes::CPU_Single_Threaded:
            single_threaded_tick(&dc, &sp, &mt);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            multi_threaded_tick();
            break;
        case SimulationModes::GPU_CUDA_mode:
            cuda_tick();
            break;
    }
}

void SimulationEngine::single_threaded_tick(EngineDataContainer * dc, SimulationParameters * sp, std::mt19937 * mt) {
    for (auto organism: dc->to_place_organisms) {
        dc->organisms.emplace_back(organism);
        for (auto &block: organism->organism_anatomy->_organism_blocks) {
            dc->simulation_grid[organism->x + block.get_pos(organism->rotation).x]
                               [organism->y + block.get_pos(organism->rotation).y].type = block.organism_block.type;

//                    if (dc->simulation_grid[organism->x + block.get_pos(organism->rotation).x]
//            [organism->y + block.get_pos(organism->rotation).y].type == BlockTypes::WallBlock) {
//                throw std::exception();
//                    }
        }

    }
    dc->to_place_organisms.clear();

    auto to_erase = std::vector<int>{};
    auto organisms_observations = std::vector<std::vector<Observation>>{};
    reserve_observations(organisms_observations, dc->organisms);

    for (auto & organism: dc->organisms) {produce_food(dc, sp, organism, *mt);}
    for (auto & organism: dc->organisms) {eat_food(dc, sp, organism);}
    for (auto & organism: dc->organisms) {apply_damage(dc, sp, organism);}

    for (int i = 0; i < dc->organisms.size(); i++) {tick_lifetime(dc, to_erase, dc->organisms[i], i);}
    for (int i = 0; i < to_erase.size(); ++i)      {erase_organisms(dc, to_erase, i);}

    for (auto & organism: dc->organisms) {get_observation(dc, organism);}
    for (int i = 0; i < dc->organisms.size(); i++) {make_decision(dc, dc->organisms[i], organisms_observations[i]);}
    //TODO VERY IMPORTANT! invalid read here
    for (auto & organism: dc->organisms) {try_make_child(dc, sp, organism, dc->to_place_organisms, mt);}
    //push_new_children(dc, dc->to_place_organisms);
    //for (auto & organism: dc->organisms) {move_organism();}
}

//Each producer will add one run of producing a food
void SimulationEngine::produce_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, std::mt19937 & mt) {
    for (int i = 0; i < organism->organism_anatomy->_producer_blocks; i++) {
        for (auto & pc: organism->organism_anatomy->_producing_space) {
            //if (check_if_out_of_boundaries(dc, organism, pc)) {continue;}
            if (dc->simulation_grid[organism->x+pc.get_pos(organism->rotation).x][organism->y+pc.get_pos(organism->rotation).y].type == BlockTypes::EmptyBlock) {
                if (std::uniform_real_distribution<float>(0, 1)(mt) < sp->food_production_probability) {
                    dc->simulation_grid[organism->x+pc.get_pos(organism->rotation).x][organism->y+pc.get_pos(organism->rotation).y].type = BlockTypes::FoodBlock;
                    break;
                }
            }
        }
    }
}

void SimulationEngine::eat_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism) {
    for (auto & pc: organism->organism_anatomy->_eating_space) {
        //TODO research why the fuck organisms reach out of bounds when they shouldn't be able to.
        if (dc->simulation_grid[organism->x+pc.get_pos(organism->rotation).x][organism->y+pc.get_pos(organism->rotation).y].type == BlockTypes::FoodBlock) {
            dc->simulation_grid[organism->x+pc.get_pos(organism->rotation).x][organism->y+pc.get_pos(organism->rotation).y].type = BlockTypes::EmptyBlock;
            organism->food_collected++;
        }
    }
}

void SimulationEngine::tick_lifetime(EngineDataContainer *dc, std::vector<int>& to_erase, Organism *organism, int organism_pos) {
    organism->lifetime++;
    if (organism->lifetime > organism->max_lifetime || organism->damage > organism->life_points) {
        for (auto & block: organism->organism_anatomy->_organism_blocks) {
            //if (check_if_out_of_boundaries(dc, organism, block)) {continue;}
            dc->simulation_grid[organism->x+block.get_pos(organism->rotation).x][organism->y+block.get_pos(organism->rotation).y].type = BlockTypes::FoodBlock;
        }
        to_erase.push_back(organism_pos);
    }
}

void SimulationEngine::erase_organisms(EngineDataContainer *dc, std::vector<int> &to_erase, int i) {
    //when erasing organism vector will decrease, so we must account for that
    delete dc->organisms[to_erase[i]-i];
    dc->organisms.erase(dc->organisms.begin() + to_erase[i] - i);
}

void SimulationEngine::apply_damage(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism) {
    for (auto & block: organism->organism_anatomy->_armor_space) {
        if (block.is_armored) {continue;}
        // It shoudln't reach as walls would stop organism
        //if (check_if_out_of_boundaries(dc, organism, block)) {continue;}
        if (dc->simulation_grid[organism->x+block.get_pos(organism->rotation).x][organism->y+block.get_pos(organism->rotation).y].type == BlockTypes::KillerBlock) {
            if (sp->one_touch_kill) {organism->damage = organism->life_points+1; break;}
            organism->damage += sp->killer_damage_amount;
        }
    }
}

void SimulationEngine::reserve_observations(std::vector<std::vector<Observation>> &observations,
                                            std::vector<Organism *> &organisms) {
    auto observations_count = std::vector<int>{};
    for (auto & organism: organisms) {
        //std::cout << "eye blocks "<< organism->organism_anatomy->_eye_blocks << "\n";
        observations_count.push_back(organism->organism_anatomy->_eye_blocks);}
    observations.reserve(observations_count.size());
    //for (auto & item: observations_count) {observations.emplace_back(std::vector<Observation>(item));}
}

void SimulationEngine::get_observation(EngineDataContainer *dc, Organism *organism) {

}

void SimulationEngine::make_decision(EngineDataContainer *dc, Organism *organism,
                                     std::vector<Observation> & organism_observations) {
    auto decision = organism->brain->get_decision(organism_observations);
}

void SimulationEngine::try_make_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                      std::vector<Organism *> &child_organisms, std::mt19937 *mt) {
    if (!organism->child_ready) {organism->child_pattern = organism->create_child(); make_child(dc, organism, mt);}
    if (organism->food_collected >= organism->child_pattern->food_needed) {
        place_child(dc, sp, organism, child_organisms, mt);}
}

//TODO probably not needed.
void SimulationEngine::make_child(EngineDataContainer *dc, Organism *organism, std::mt19937 * mt) {
    organism->rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
    organism->child_ready = true;
}

void SimulationEngine::place_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                   std::vector<Organism *> &child_organisms, std::mt19937 *mt) {
    auto to_place = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
    Rotation rotation;
    if (sp->rotation_enabled) {
        rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
    } else {
        rotation = Rotation::UP;
    }
    auto distance = std::uniform_int_distribution<int>(0, 3)(*mt);
    //UP - min_y,
    //LEFT - min_x
    //DOWN - max_y
    //RIGHT - max_x

    auto min_y = 0;
    auto min_x = 0;
    auto max_y = 0;
    auto max_x = 0;

    //a width and height of an organism can only change by one, so to be safe, the distance between organisms = size_of_base_organism + 1
    switch (to_place) {
        case Rotation::UP:
            for (auto & block: organism->organism_anatomy->_organism_blocks) {
                if (block.get_pos(organism->rotation).y < min_y) {min_y = block.get_pos(organism->rotation).y;}
            }
            min_y -= distance;
            break;
        case Rotation::LEFT:
            for (auto & block: organism->organism_anatomy->_organism_blocks) {
                if (block.get_pos(organism->rotation).x < min_x) {min_x = block.get_pos(organism->rotation).x;}
            }
            min_x -= distance;
            break;
        case Rotation::DOWN:
            for (auto & block: organism->organism_anatomy->_organism_blocks) {
                if (block.get_pos(organism->rotation).x > max_y) {max_y = block.get_pos(organism->rotation).y;}
            }
            max_y += distance;
            break;
        case Rotation::RIGHT:
            for (auto & block: organism->organism_anatomy->_organism_blocks){
                if (block.get_pos(organism->rotation).x > max_x) {max_x = block.get_pos(organism->rotation).x;}
            }
            max_x += distance;
            break;
    }

    organism->child_pattern->x = organism->x + min_x + max_x;
    organism->child_pattern->y = organism->y + min_y + max_y;
    organism->child_pattern->rotation = rotation;

    //checking, if there is space for a child
    for (auto & block: organism->child_pattern->organism_anatomy->_organism_blocks) {
        if (check_if_out_of_boundaries(dc, organism->child_pattern, block)) {return;}

        if (dc->simulation_grid[organism->child_pattern->x + block.get_pos(organism->child_pattern->rotation).x]
                               [organism->child_pattern->y + block.get_pos(organism->child_pattern->rotation).y].type != BlockTypes::EmptyBlock) {return;}
    }
    //if (dc->simulation_grid[organism->x+pc.get_pos(organism->rotation).x][organism->y+pc.get_pos(organism->rotation).y].type == BlockTypes::EmptyBlock)

    child_organisms.emplace_back(organism->child_pattern);
    organism->food_collected -= organism->child_pattern->food_needed;
    organism->child_pattern = nullptr;
    organism->child_ready = false;
}

void SimulationEngine::tick_of_single_thread() {

}

void SimulationEngine::multi_threaded_tick() {
    for (auto & thread :threads) {
        thread.work();
    }

    for (auto & thread: threads) {
        thread.finish();
    }
}

void SimulationEngine::cuda_tick() {

}

void SimulationEngine::kill_threads() {
    if (!threads.empty()) {
        for (auto & thread: threads) {
            thread.stop_work();
        }
        threads.clear();
    }
}

void SimulationEngine::build_threads() {
    kill_threads();
    threads.reserve(cp.num_threads);

    thread_points.clear();
    thread_points = Linspace<int>()(0, dc.simulation_width, cp.num_threads+1);

    for (int i = 0; i < cp.num_threads; i++) {
        threads.emplace_back(&dc, thread_points[i], 0, thread_points[i+1], dc.simulation_height);
    }
    cp.build_threads = false;
}

// if any is true, then check fails, else check succeeds
//template<typename T>
bool SimulationEngine::check_if_out_of_boundaries(EngineDataContainer *dc, Organism *organism, BaseSerializedContainer &block) {
    return (organism->x + block.get_pos(organism->rotation).x < 0 ||
            organism->x + block.get_pos(organism->rotation).x > dc->simulation_width -1 ||
            organism->y + block.get_pos(organism->rotation).y < 0 ||
            organism->y + block.get_pos(organism->rotation).y > dc->simulation_height-1);
}
