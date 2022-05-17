//
// Created by spaceeye on 16.05.2022.
//

#include "SimulationEnginePartialMultiThread.h"

void SimulationEnginePartialMultiThread::partial_multi_thread_tick(EngineDataContainer *dc,
                                                                   EngineControlParameters *cp,
                                                                   OrganismBlockParameters *bp,
                                                                   SimulationParameters *sp) {
    auto to_place_thread_points = std::vector<int>{};
    calculate_threads_points(dc->to_place_organisms.size(), dc->threads.size(), to_place_thread_points);

    start_stage(PartialSimulationStage::PlaceOrganisms, to_place_thread_points);

//    for (auto organism: dc->to_place_organisms) {
//        dc->organisms.emplace_back(organism);
//        for (auto &block: organism->organism_anatomy->_organism_blocks) {
//            dc->single_thread_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
//            [organism->y + block.get_pos(organism->rotation).y].type = block.organism_block.type;
//        }
//    }
//    dc->to_place_organisms.clear();

    auto simulation_organism_thread_points = std::vector<int>{};
    calculate_threads_points(dc->organisms.size(), dc->threads.size(), simulation_organism_thread_points);

    auto to_erase = std::vector<int>{};
    auto organisms_observations = std::vector<std::vector<Observation>>{};
    reserve_observations(organisms_observations, dc->organisms);

    start_stage(PartialSimulationStage::ProduceFood, simulation_organism_thread_points);
    start_stage(PartialSimulationStage::EatFood, simulation_organism_thread_points);
    start_stage(PartialSimulationStage::ApplyDamage, simulation_organism_thread_points);
    start_stage(PartialSimulationStage::TickLifetime, simulation_organism_thread_points);
    start_stage(PartialSimulationStage::GetObservations, simulation_organism_thread_points);
    start_stage(PartialSimulationStage::MakeDecision, simulation_organism_thread_points);
    //start_stage(PartialSimulationStage::TryMakeChild, simulation_organism_thread_points);


}

void SimulationEnginePartialMultiThread::build_threads(EngineDataContainer &dc, EngineControlParameters &cp) {
    kill_threads(dc);
    dc.threads.reserve(cp.num_threads);

    dc.thread_points.clear();
    dc.thread_points = Linspace<int>()(0, dc.simulation_width, cp.num_threads+1);

    for (int i = 0; i < cp.num_threads; i++) {
        dc.threads.emplace_back(&dc);
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

void SimulationEnginePartialMultiThread::place_organisms(EngineDataContainer *dc) {

}

void
SimulationEnginePartialMultiThread::produce_food(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                                 std::mt19937 &mt) {

}

void
SimulationEnginePartialMultiThread::eat_food(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism) {

}

void SimulationEnginePartialMultiThread::tick_lifetime(EngineDataContainer *dc, std::vector<int> &to_erase,
                                                       Organism *organism, int organism_pos) {

}

void SimulationEnginePartialMultiThread::erase_organisms(EngineDataContainer *dc, std::vector<int> &to_erase, int i) {

}

void SimulationEnginePartialMultiThread::apply_damage(EngineDataContainer *dc, SimulationParameters *sp,
                                                      Organism *organism) {

}

void SimulationEnginePartialMultiThread::reserve_observations(std::vector<std::vector<Observation>> &observations,
                                                              std::vector<Organism *> &organisms) {

}

void SimulationEnginePartialMultiThread::get_observations(EngineDataContainer *dc, std::vector<Organism *> &organisms,
                                                          std::vector<std::vector<Observation>> &organism_observations) {

}

void SimulationEnginePartialMultiThread::rotate_organism(EngineDataContainer *dc, Organism *organism,
                                                         BrainDecision decision) {

}

void
SimulationEnginePartialMultiThread::move_organism(EngineDataContainer *dc, Organism *organism, BrainDecision decision) {

}

void
SimulationEnginePartialMultiThread::make_decision(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                                  std::vector<Observation> &organism_observations) {

}

void
SimulationEnginePartialMultiThread::do_decision(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                                std::vector<Observation> &organism_observations) {

}

void SimulationEnginePartialMultiThread::try_make_child(EngineDataContainer *dc, SimulationParameters *sp,
                                                        Organism *organism, std::vector<Organism *> &child_organisms,
                                                        std::mt19937 *mt) {

}

void SimulationEnginePartialMultiThread::make_child(EngineDataContainer *dc, Organism *organism, std::mt19937 *mt) {

}

void
SimulationEnginePartialMultiThread::place_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                                std::vector<Organism *> &child_organisms, std::mt19937 *mt) {

}

bool SimulationEnginePartialMultiThread::check_if_out_of_boundaries(EngineDataContainer *dc, Organism *organism,
                                                                    BaseSerializedContainer &block, Rotation rotation) {
    return false;
}

void SimulationEnginePartialMultiThread::calculate_threads_points(int num_organisms, int num_threads,
                                                                  std::vector<int> &thread_points) {

}

void SimulationEnginePartialMultiThread::thread_tick(PartialSimulationStage stage, int start_pos, int end_pos) {

}

void SimulationEnginePartialMultiThread::start_stage(PartialSimulationStage stage, std::vector<int> &thread_points) {

}
