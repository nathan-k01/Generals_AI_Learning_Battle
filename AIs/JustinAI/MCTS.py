"""Open-loop MCTS with sampled hidden states and UCB selection."""
import math
from .Evaluation import evaluate, move_score

class Node:
    def __init__(self):
        self.visits=0; self.total=0.; self.children={}

def policy(s,p,sim,rng):
    return max(sim.actions(s,p),key=lambda m:move_score(s,p,m,sim)+rng.uniform(0,2))

def search(sim,player,legal,rng,iterations=48,depth=6):
    if not legal: return None
    root=Node()
    for _ in range(iterations):
        s=sim.sample(rng); initial=evaluate(s,player,sim); node=root; path=[root]
        for d in range(depth):
            if player not in s.active or len(s.active)<=1 or s.turn>=sim.obs['max_turns']: break
            options=legal if d==0 else list(sim.actions(s,player))
            options=sorted(options,key=lambda m:move_score(s,player,m,sim),reverse=True)[:8]
            fresh=[m for m in options if m not in node.children]
            if fresh:
                m=fresh[0]; child=Node(); node.children[m]=child
            else:
                m=max(options,key=lambda a:node.children[a].total/node.children[a].visits+1.4*math.sqrt(math.log(max(1,node.visits))/node.children[a].visits)); child=node.children[m]
            moves={p:policy(s,p,sim,rng) for p in s.active if p!=player}; moves[player]=m
            s=sim.step(s,moves); node=child; path.append(node)
            if fresh:
                for _ in range(d+1,depth):
                    if player not in s.active or len(s.active)<=1 or s.turn>=sim.obs['max_turns']: break
                    s=sim.step(s,{p:policy(s,p,sim,rng) for p in s.active})
                break
        reward=math.tanh((evaluate(s,player,sim)-initial)/25)
        for n in path: n.visits+=1; n.total+=reward
    return max(root.children,key=lambda m:(root.children[m].visits,root.children[m].total/root.children[m].visits)) if root.children else legal[0]
