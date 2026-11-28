import random
from .Simulator import Simulator
from .MCTS import search

class AI:
    def __init__(self,player_id=2,iterations=48,depth=6,seed=None):
        if iterations<1 or depth<1: raise ValueError('Positive search budget required')
        self.player_id=player_id; self.iterations=iterations; self.depth=depth; self.rng=random.Random(seed)

    def choose_action(self,environment):
        obs=environment.get_observation(self.player_id)
        actions=environment.get_legal_actions(self.player_id)
        if not actions: return None
        sim=Simulator(obs)
        def encode(a):
            if a.is_pass: return None
            x,y=a.source; u,v=a.destination
            return (y*sim.w+x,v*sim.w+u,a.half)
        mapping={encode(a):a for a in actions}
        return mapping[search(sim,self.player_id,list(mapping),self.rng,self.iterations,self.depth)]

    def take_turn(self,environment):
        if environment.is_waiting_for_player(self.player_id):
            action=self.choose_action(environment)
            if action is not None: return environment.submit_action(self.player_id,action)
