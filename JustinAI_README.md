# JustinAI — Monte Carlo Tree Search

Independent classical search agent for Generals_AI_Learning_Battle. No neural network, training, or teammate AI imports.

## Installation
Copy the `AIs/JustinAI` directory and `test_justin_ai.py` into the existing repository, preserving paths. From the repository root run `python -m pip install pygame` if needed.

## Test
`python test_justin_ai.py --games 3 --turns 100 --iterations 24`

For full-length matches use `--turns 400`. The engine declares a draw at its turn limit, so territory advantage does not automatically count as a win. Test against trained teammates separately. Increase iterations for stronger search at greater runtime cost.

## Integrate into main.py
Add `from AIs.JustinAI.AI import AI as JustinAI` near imports. In `evaluate_individual`, after creating `kal_ai`, add `justin_ai = JustinAI(player_id=2)`. Replace `random_ai_move(env, 2)` with `justin_ai.take_turn(env)`. Keep the existing training logic intact. Each evaluation gets its own agent. Do not share one instance across matches.

## Algorithm
The agent reads only its own observation and legal actions. Each iteration samples a plausible hidden board, selects a move using upper confidence bounds (UCB), expands a node, simulates remaining turns with a heuristic rollout policy, and propagates the reward back up the tree. The most visited root move is returned. Search is open-loop: nodes represent action sequences, not exact hidden states. Each simulated turn resolves all active players' moves together, then applies growth. The evaluation rewards territory, army size, castles, frontier strength and general safety. It penalizes strong opponents. The rollout policy is independent of the other agents.

## Limitations
Hidden enemy positions and army sizes are estimates, not learned beliefs. Candidate pruning keeps eight promising moves per node. The simulator approximates the shared engine and does not model all fog-of-war knowledge updates. The default search budget is 48 iterations and six turns. No win rate against trained agents is claimed. Python syntax and synthetic smoke tests were checked; live repository tests must be run locally because cloning the repository in the execution environment failed due to DNS.

## Reference
Browne et al., A Survey of Monte Carlo Tree Search Methods, IEEE TCIAIG (2012), DOI: 10.1109/TCIAIG.2012.2186810.
