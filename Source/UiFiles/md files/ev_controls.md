# Evolution Controls
These are the explanations for some evolution options
- **"Lifespan multiplier"** - Multiplicator of the sum of "Lifetime Weight" of each block of an organism_index.
- **"Anatomy mutation rate mutation step"** - An amount by which a mutation rate of an organism_index can increase or decrease
- **"Anatomy mutation rate delimiter"** - A parameter which controls whatever organism_index's anatomy mutation rate will be biased to increase or decrease. If >0.5 then the rate will increase, if <0.5 then the rate will decrease, if == 0.5 then no bias.
- **"Brain mutation rate mutation step"** - The same as "Anatomy mutation rate mutation step"
- **"Brain mutation rate delimiter"** - The same as "Anatomy mutation rate delimiter"
- **"Fix reproduction distance"** - Will make reproduction distance always equal to "Min reproduction distance" during reproduction.
- **"Organism's self cells block sight"** - If disabled, organism_index can see through itself. If enabled, the eye that points to the cell that belongs to itself will return "Empty block" as observation.
- **"Set fixed move range"** - Will force organisms to use "Min move range" and will make child move ranges equal to parent during reproduction.
- **"Move range delimiter"** - The same as Anatomy mutation rate delimiter”
- **"Failed reproduction eats food"** - If disabled, then the food will be deduced from parent organism_index only after successful reproduction.
- **"Rotate every move tick"** - Will make organisms rotate every time they move. If disabled, then they will rotate only at the end of move range.
- **"Simplified food production"** - Will try to produce food for each space that can produce food.
- **"Eat first, then produce food"** - If disabled, will produce food first and only then eat.
- **"Use new child position calculator"** - New child position calculator calculating position of a child by first calculating the coordinates of edges of parent and child + distance. For example, if the chosen reproduction direction is up, then calculator will calculate the uppermost y cell coordinate of a parent, the bottom y cell coordinate of a child + distance. That way, the child organism_index will never appear inside a parent organism_index. The old child position calculator however calculates only the edge coordinates of a parent organism_index + distance, allowing child organisms to appear inside parent, with the side effect of organisms being unable to reproduce if the reproducing distance is less than (height or width)/2 (depending on child organism_index rotation and chosen reproductive direction)
- **"Check if path is clear"** - If enabled, will check for each cell of a child organism_index if there is an obstruction in the way (except for parents cells), like a wall or a food if "Food blocks reproduction" is enabled. If there is, then the parent organism_index will not reproduce. If disabled, the child will just check if there is a space for itself at the end coordinated, but it will introduce some behaviours such as child organisms hopping though walls if they are thin enough.

"Organism cell parameters modifiers" - modifiers for each cell of all organisms.
- **"Life point"** - The amount of life points this cell will give to organism_index
- **"Lifetime weight"** - The amount of lifetime this cell will give to organism_index
- **"Chance Weight"** - Controls how likely this cell will be picked during reproduction compared to others. If 0, the cell will never get picked.
