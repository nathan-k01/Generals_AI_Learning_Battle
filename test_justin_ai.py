import argparse, random, time
from Game.generals_game import GeneralsEnvironment
from AIs.JustinAI.AI import AI

def run(seed, turns, iterations):
    env=GeneralsEnvironment(seed=seed,max_turns=turns)
    rng=random.Random(seed); ai=AI(2,iterations=iterations,seed=seed)
    start=time.perf_counter()
    while not env.game_over:
        for p in env.get_pending_players():
            if p==2:
                action=ai.choose_action(env)
                assert action in env.get_legal_actions(p)
            else: action=rng.choice(env.get_legal_actions(p))
            assert env.submit_action(p,action)['accepted']
    print(f'Seed {seed}: winner={env.winner}, turns={env.turn}, Justin stats={env.get_stats(2)}, seconds={time.perf_counter()-start:.2f}')
    return env.winner

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--games',type=int,default=3); p.add_argument('--turns',type=int,default=100); p.add_argument('--iterations',type=int,default=24); a=p.parse_args()
    results=[run(i,a.turns,a.iterations) for i in range(a.games)]
    print(f'Wins: {results.count(2)}, draws: {results.count(None)}, losses: {sum(x in (0,1) for x in results)}')
