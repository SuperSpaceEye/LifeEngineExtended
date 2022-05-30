//
// Created by spaceeye on 16.05.2022.
//

#include "SimulationEnginePartialMultiThread.h"

void SimulationEnginePartialMultiThread::partial_multi_thread_tick(EngineDataContainer *dc,
                                                                   EngineControlParameters *cp,
                                                                   OrganismBlockParameters *bp,
                                                                   SimulationParameters *sp,
                                                                   boost::mt19937 *mt) {
    for (auto & organism: dc->to_place_organisms) {place_organism(dc, organism); dc->organisms.emplace_back(organism);}
    dc->to_place_organisms.clear();

    auto simulation_organism_thread_points = std::vector<std::vector<int>>{};
    calculate_threads_points(dc->organisms.size(), dc->threads.size(), simulation_organism_thread_points);

    start_stage(dc, PartialSimulationStage::EatFood, simulation_organism_thread_points);
    start_stage(dc, PartialSimulationStage::ProduceFood, simulation_organism_thread_points);
    if (sp->killer_damage_amount > 0) {
    start_stage(dc, PartialSimulationStage::ApplyDamage, simulation_organism_thread_points);}
    start_stage(dc, PartialSimulationStage::TickLifetime, simulation_organism_thread_points);

    erase_organisms(dc, dc->threaded_to_erase);

    simulation_organism_thread_points = std::vector<std::vector<int>>{};
    calculate_threads_points(dc->organisms.size(), dc->threads.size(), simulation_organism_thread_points);

    SimulationEngineSingleThread::reserve_observations(dc->threaded_organisms_observations, dc->organisms, sp);
    start_stage(dc, PartialSimulationStage::GetObservations, simulation_organism_thread_points);

    start_stage(dc, PartialSimulationStage::ThinkDecision, simulation_organism_thread_points);
    for (int i = 0; i < dc->organisms.size(); i++)  {SimulationEngineSingleThread::make_decision(dc, sp, dc->organisms[i], mt);}

    for (auto & organism: dc->organisms) {SimulationEngineSingleThread::try_make_child(dc, sp, organism, dc->to_place_organisms, mt);}
}

void SimulationEnginePartialMultiThread::build_threads(EngineDataContainer &dc, EngineControlParameters &cp,
                                                       SimulationParameters &sp) {
    kill_threads(dc);
    dc.threads.reserve(cp.num_threads);
    dc.threaded_to_erase.clear();
    dc.threaded_to_erase.reserve(cp.num_threads);
    for (int i = 0; i < cp.num_threads; i++) {
        dc.threaded_to_erase.emplace_back(std::vector<int>{});
    }

    for (int i = 0; i < cp.num_threads; i++) {
        dc.threads.emplace_back(eager_worker_partial{&dc, &sp, i});
    }
    cp.build_threads = false;
}

void SimulationEnginePartialMultiThread::kill_threads(EngineDataContainer &dc) {
    if (!dc.threads.empty()) {
        for (auto & thread: dc.threads) {
            thread.stop_work();
        }
        dc.threads.clear();
    }
}

void SimulationEnginePartialMultiThread::calculate_threads_points(int num_points, int num_threads,
                                                                  std::vector<std::vector<int>> &thread_points) {
    if (num_points < num_threads) {
        for (int i = 0; i < num_points; i++) {
            thread_points.push_back(std::vector<int>{i, i});
        }
        return;
    }

    for (int i = 0; i < num_threads; i++) {
        thread_points.push_back(std::vector<int>{i * (num_points/num_threads), (i+1) * (num_points/num_threads)});
    }
    thread_points[thread_points.size()-1][1] = num_points;
}

void SimulationEnginePartialMultiThread::start_stage(EngineDataContainer *dc, PartialSimulationStage stage,
                                                     std::vector<std::vector<int>> &thread_points) {
    for (int i = 0; i < thread_points.size(); i++) {
        dc->threads[i].work(stage, thread_points[i][0], thread_points[i][1]);
    }

    for (auto & thread: dc->threads) {
        //while (thread.has_work){}
        thread.finish();
    }

    if (stage == PartialSimulationStage::PlaceOrganisms) {
        for (auto organism: dc->to_place_organisms) {
            dc->organisms.emplace_back(organism);
        }
        dc->to_place_organisms.clear();
    }

    if (stage == PartialSimulationStage::GetObservations) {
        dc->threaded_organisms_observations.clear();
    }
}

void SimulationEnginePartialMultiThread::place_organism(EngineDataContainer *dc, Organism *organism) {
    for (auto &block: organism->organism_anatomy->_organism_blocks) {
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].type = block.type;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].neighbors = block.get_rotated_block_neighbors(organism->rotation);
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].rotation = block.rotation;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].organism = organism;
    }
}

//void SimulationEnginePartialMultiThread::produce_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, boost::mt19937 & mt) {
//    if (organism->organism_anatomy->_producer_blocks <= 0) {return;}
//    if (organism->organism_anatomy->_mover_blocks > 0 && !sp->movers_can_produce_food) {return;}
//    if (organism->lifetime % sp->produce_food_every_n_life_ticks != 0) {return;}
//
//    for (int i = 0; i < organism->organism_anatomy->_producer_blocks; i++) {
//        for (auto & pc: organism->organism_anatomy->_producing_space) {
//            //TODO locking here?
//            if (dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type == EmptyBlock) {
//                if (std::uniform_real_distribution<float>(0, 1)(mt) < sp->food_production_probability) {
//                    dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type = FoodBlock;
//                    break;
//                }
//            }
//        }
//    }
//}

void SimulationEnginePartialMultiThread::eat_food(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism) {
    for (auto & pc: organism->organism_anatomy->_eating_space) {
        while (dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].locked) {}
        dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].locked = true;
        if (dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type == FoodBlock) {
            dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type = EmptyBlock;
            organism->food_collected++;
        }
        dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].locked = false;
    }
}

void SimulationEnginePartialMultiThread::tick_lifetime(EngineDataContainer *dc, std::vector<std::vector<int>> &to_erase,
                                                       Organism *organism, int thread_num, int organism_pos) {
    organism->lifetime++;
    if (organism->lifetime > organism->max_lifetime || organism->damage > organism->life_points) {
        for (auto & block: organism->organism_anatomy->_organism_blocks) {
            dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y].type = FoodBlock;
            dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y].organism = nullptr;
        }
        to_erase[thread_num].emplace_back(organism_pos);
    }
}

void SimulationEnginePartialMultiThread::erase_organisms(EngineDataContainer * dc, std::vector<std::vector<int>>& to_erase) {
    int i = 0;
    for (auto & thread_delete: dc->threaded_to_erase) {
        for (auto & ii: thread_delete) {
                delete dc->organisms[ii-i];
                dc->organisms.erase(dc->organisms.begin() + ii - i);
                i++;
        }
        thread_delete.clear();
    }
}

void SimulationEnginePartialMultiThread::get_observations(EngineDataContainer *dc, Organism *&organism,
                                                          std::vector<std::vector<Observation>> &organism_observations,
                                                          SimulationParameters *sp, int organism_num) {
    organism = dc->organisms[organism_num];
    if (organism->organism_anatomy->_eye_blocks <= 0 || organism->organism_anatomy->_mover_blocks <=0) {return;}
    if (organism->move_counter != 0) {return;}
    auto eye_i = -1;
    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        if (block.type != BlockTypes::EyeBlock) {continue;}
        eye_i++;
        auto pos_x = organism->x + block.get_pos(organism->rotation).x;
        auto pos_y = organism->y + block.get_pos(organism->rotation).y;
        // getting global rotation on a simulation grid
        auto block_rotation = block.get_block_rotation_on_grid(organism->rotation);

        auto offset_x = 0;
        auto offset_y = 0;

        switch (block_rotation) {
            case Rotation::UP:
                offset_y = -1;
                break;
            case Rotation::LEFT:
                offset_x = -1;
                break;
            case Rotation::DOWN:
                offset_y = 1;
                break;
            case Rotation::RIGHT:
                offset_x = 1;
                break;
        }

        auto last_observation = Observation{EmptyBlock, 0, block.rotation};

        for (int i = 1; i < sp->look_range; i++) {
            pos_x += offset_x;
            pos_y += offset_y;

            if (SimulationEngineSingleThread::check_if_out_of_bounds(dc, pos_x, pos_y)) {break;}

            last_observation.type = dc->CPU_simulation_grid[pos_x][pos_y].type;
            last_observation.distance = i;

            if (last_observation.type == BlockTypes::WallBlock) {break;}
            if (last_observation.type == BlockTypes::FoodBlock) {break;}
            if (last_observation.type != BlockTypes::EmptyBlock) {
                if (!sp->organism_self_blocks_block_sight) {
                    if(dc->CPU_simulation_grid[pos_x][pos_y].organism == organism) {
                        continue;
                    }
                }
                break;
            }
        }
        organism_observations[organism_num][eye_i] = last_observation;
    }
}

void SimulationEnginePartialMultiThread::thread_tick(PartialSimulationStage stage, EngineDataContainer *dc,
                                                SimulationParameters *sp, boost::mt19937 *mt, int start_pos,
                                                int end_pos, int thread_num) {
    switch (stage) {
        case PartialSimulationStage::PlaceOrganisms:
            for (int i = start_pos; i < end_pos; i++) {
                place_organism(dc, dc->to_place_organisms[i]);
            }
            break;
        case PartialSimulationStage::ProduceFood:
            for (int i = start_pos; i < end_pos; i++) {
                SimulationEngineSingleThread::produce_food(dc, sp, dc->organisms[i], *mt);
            }
            break;
        case PartialSimulationStage::EatFood:
            for (int i = start_pos; i < end_pos; i++) {
                SimulationEnginePartialMultiThread::eat_food(dc, sp, dc->organisms[i]);
            }
            break;
        case PartialSimulationStage::ApplyDamage:
            for (int i = start_pos; i < end_pos; i++) {
                SimulationEngineSingleThread::apply_damage(dc, sp, dc->organisms[i]);
            }
            break;
        case PartialSimulationStage::TickLifetime:
            for (int i = start_pos; i < end_pos; i++) {
                SimulationEnginePartialMultiThread::tick_lifetime(dc, dc->threaded_to_erase, dc->organisms[i], thread_num, i);
            }
            break;
        case PartialSimulationStage::GetObservations:
            for (int i = start_pos; i < end_pos; i++) {
                SimulationEnginePartialMultiThread::get_observations(dc, dc->organisms[i],
                                                                     dc->threaded_organisms_observations, sp,
                                                                     i);
            }
            break;
        case PartialSimulationStage::ThinkDecision:
            for (int i = start_pos; i < end_pos; i++) {
                dc->organisms[i]->think_decision(dc->threaded_organisms_observations[i], mt);
            }
            break;
    }
}
