# TheLifeEngineCpp
C++ implementation of https://github.com/MaxRobinsonTheGreat/LifeEngine.

This version is not fully implemented, and will be finished in the future.

To build run CMake. You will need Qt5, boost.

## Explanation of The Life Engine cpp setting.
option | type | constraints | desciption

#### World Controls
- Max organisms | int | any int | Will limit the amount of organisms in the simulation by preventing reproduction of new organism, untill there is available space for them. If set to 0, will stop reproduction of organims. If set to any number <=-1, will disable the limit of organims.

#### Evolution Controls
- Food production probability | float | 0<x<=1 | This option controls probablity of each producing cell to produce a food cell around it.
- Produce food every n life ticks | int | x>0 | On every n tick of organism's lifetime it will try to produce food.
- Lifespan multiplier | int | x>0 |
- Look range | int | x>0 |
- Auto food drop rate | int | x>=0 | Function is not yet implemented
- Extra reproduction cost | int | any | Function is not yet implemented
- Global anatomy mutation rate | float | 0<=x<=1 | If "Use evolved anatomy mutation rate" is not checked, will use this mutation rate instead.
- Anatomy mutation rate delimiter | float | 0<=x<=1 | A delimiter between increasing evolved mutation rate, or decreasing. If x>0.5 , then it will bias organisms to increase mutation rate, if x<0.5, then it will bias organisms to decrease it. If x==0.5, then there will be no bias.
- Global brain mutation rate | float | 0<=x<=1 | The same as Global anatomy mutation rate.
- Brain mutation rate delimiter | float | 0<=x<=1 | The same as Anatomy mutation rate delimiter.
- Killer damage amount | int | x>=0 | If x==0, then simulation will skip stage of applying damage.
- add/change/remove cell | int | x>=0 | Weight of every choice when deciding on type of mutation.
- Min/Max reproducing distance | int | x>=1, Min distance can't be more than Max, and Max distance can't be less than Min |
- Fix reproducing distance | If not checked, the organims will try to place a child anywere between min/max distance of itself. If checked, organism will always try to place a child min distance from himself.
- Organism's self cells block sight | If checked, an organism's self cells will block it's eye cell sight.

#### Settings
- Float number precision | int | x>=0 | A cosmetic option, that will determine how many numbers after a point, displayed float numbers will show.
