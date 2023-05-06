#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <boost/range/combine.hpp>

#include "Organism/Organism.h"

//using SOBC = SerializedOrganismBlockContainer;
//using BT = BlockTypes;
//using Catch::Matchers::WithinRel;
//
//std::shared_ptr<Organism> make_organism() {
//    auto anatomy = Anatomy();
//    anatomy.set_many_blocks(std::vector<SOBC>{
//        SOBC{BT::MouthBlock, Rotation::UP, 0, 0},
//        SOBC{BT::ProducerBlock, Rotation::UP, -1, -1},
//        SOBC{BT::ProducerBlock, Rotation::UP, 1, 1},
//    });
//
//    auto occ = OrganismConstructionCode();
//    occ.set_code(std::vector<OCCInstruction>{OCCInstruction::SetBlockMouth,
//                                             OCCInstruction::ShiftUpLeft, OCCInstruction::SetBlockProducer,
//                                             OCCInstruction::ShiftDownRight, OCCInstruction::SetBlockProducer});
//
//    auto sp = new SimulationParameters();
//    auto bp = new OrganismBlockParameters();
//    auto occp = new OCCParameters();
//    auto occl = new OCCLogicContainer();
//
//    return std::make_shared<Organism>(
//            0, 0, Rotation::UP,
//            anatomy, Brain(), occ, sp, bp, occp, occl, 5, 0.05, 0.1, true
//            );
//}
//
//void cleanup_organism(std::shared_ptr<Organism> & organism) {
//    delete organism->sp;
//    delete organism->bp;
//    delete organism->occp;
//    delete organism->occl;
//}
//
//float o_calculate_mass(const Organism & o) {
//    int max_point = o.sp->growth_of_organisms ? o.size : o.anatomy.organism_blocks.size();
//    return std::accumulate(o.anatomy.organism_blocks.begin(), o.anatomy.organism_blocks.begin()+max_point, 0,
//          [&](auto sum, auto & item){return sum + o.bp->pa[int(item.type)-1].food_cost;});
//}
//
//float calculate_max_life(const Organism & o) {
//    int max_point = o.anatomy.organism_blocks.size();
//    if (o.sp->growth_of_organisms) {max_point = o.size;}
//    return std::accumulate(o.anatomy.organism_blocks.begin(), o.anatomy.organism_blocks.begin()+max_point,0,
//    [&](auto sum, auto & item){return sum + o.bp->pa[int(item.type)-1].life_point_amount;});
//}
//
//int calculate_organism_lifetime(const Organism & o) {
//    int max_point = o.anatomy.organism_blocks.size();
//    if (o.sp->growth_of_organisms) {max_point = o.size;}
//    float lifetime_weights = std::accumulate(o.anatomy.organism_blocks.begin(), o.anatomy.organism_blocks.begin()+max_point, 0,
//    [&](auto sum, auto & item){return sum + o.bp->pa[int(item.type)-1].lifetime_weight;});
//
//    return static_cast<int>(lifetime_weights * o.sp->lifespan_multiplier);
//}
//
//float calculate_food_needed(const Organism & o) {
//    float food_needed = o.sp->extra_reproduction_cost + o.sp->extra_mover_reproductive_cost * (o.anatomy.c[BT::MoverBlock] > 0);
//
//    int max_point = o.anatomy.organism_blocks.size();
//    if (o.sp->growth_of_organisms) {max_point = o.size;}
//    return std::accumulate(o.anatomy.organism_blocks.begin(), o.anatomy.organism_blocks.begin()+max_point, food_needed,
//    [&](auto sum, auto & item){return sum + o.bp->pa[int(item.type)-1].food_cost;});
//}
//
//static void check_preinit(std::shared_ptr<Organism> &o) {
//    if (!o->sp->growth_of_organisms) {
//        REQUIRE(o->c == o->anatomy.c);
//
//        auto &blocks = o->anatomy.organism_blocks;
//        auto view = o->get_organism_blocks_view();
//
//        REQUIRE(blocks.size() == view.size());
//        for (const auto &[block1, block2]: boost::combine(blocks, view)) {
//            REQUIRE(block1 == block2);
//        }
//
//        REQUIRE(o->is_adult);
//        REQUIRE(o->size == (uint32_t) -1);
//    } else {
//        auto tc = make_anatomy_counters();
//        for (int i = 0; i < std::min<int64_t>(
//                o->sp->starting_organism_size, o->anatomy.organism_blocks.size()); i++){
//            o->c[o->anatomy.organism_blocks[i].type]++;
//        }
//        REQUIRE(o->c == tc);
//        REQUIRE(o->size == std::min<int>(o->sp->starting_organism_size, o->anatomy.organism_blocks.size()));
//        REQUIRE(o->is_adult == (o->size == o->anatomy.organism_blocks.size()));
//    }
//
//    REQUIRE_THAT(o->mass, WithinRel(o_calculate_mass(*o), 0.01));
//    REQUIRE(o->lifetime == 0);
//    REQUIRE(o->damage == 0);
//}
//
//TEST_CASE("Organism initialization no growth", "[organism][no_growth]") {
//    auto o = make_organism();
//    auto & sp = *o->sp;
//
//    sp.growth_of_organisms = false;
//
//    o->pre_init();
//    o->init_values();
//
//    check_preinit(o);
//
//    REQUIRE_THAT(o->life_points, WithinRel(calculate_max_life(*o), 0.01));
//    REQUIRE(o->max_lifetime == calculate_organism_lifetime(*o));
//    REQUIRE_THAT(o->food_needed, WithinRel(calculate_food_needed(*o), 0.01));
//
//    cleanup_organism(o);
//}
