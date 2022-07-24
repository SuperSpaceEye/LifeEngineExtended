These are the explanations for some evolution options
- "Lifespan multiplier" - Multiplicator of the sum of "Lifetime Weight" of each block of an organism.
- "Anatomy mutation rate mutation step" - An amount by which a mutation rate of an organism can increase or decrease
- "Anatomy mutation rate delimiter" - A parameter which controls whetever organism's anatomy mutation rate will be biased to increase or decrease. If >0.5 then the rate will increase, if <0.5 then the rate will decrease, if == 0.5 then no bias.
- "Brain mutation rate mutation step" - The same as "Anatomy mutation rate mutation step"
- "Brain mutation rate delimiter" - The same as "Anatomy mutation rate delimiter"
- "Fix reproduction distance" - Will make reproduction distance always equal to "Min repdoduction distance" during reproduction.
- "Organism's self cells block sight"- If disabled, organism can see through itself. If enabled, the eye that points to the cell that belongs to itself will return "Empty block" as observation.
- "Set fixed move range" - Will force organisms to use "Min move range" and will make child move ranges equal to parent during reproduction.
- "Move range delimiter" - The same as Anatomy mutation rate delimiter”
- "Failed reproduction eats food" - If disabled, then the food will be deduced from parent organism only after succesfull reproduction.
- "Rotate every move tick" - Will make organisms rotate every time they move. If disabled, then they will rotate only at the end of move range.
- "Simplified food production" - Will try to produce food for each space that can produce food.
- "Eat first, then produce food" - If disabled, will produce food first and only then eat.
- "Use new child position calculator" - If disabled, will use old buggy version which shouldn't be used.
- "Check if path is clear" - If enabled, will check for each cell (except for parents) of a child organism if there is an obstruction in the way, like a wall or a food if "Food blocks reproduction" is enabled. If there is then the parent organism will not reproduce. If disabled, the child will just check if there is a space for itself at the end coordinated, but it will introduce some behaviours such as child organisms hopping though walls if they are thin enogh.

"Organism cell parameters modifiers" - modifiers for each cell of all organisms.
- "Life point" - The amount of life points this cell will give to organism
- "Lifetime weight" - The amount of lifetime this cell will give to organism
- "Chance Weight" - Controls how likely this cell will be picked during reproduction compared to others. If 0, the cell will never get picked.