"""Approximate simultaneous-turn simulator using only public observations."""
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    board: tuple
    turn: int
    active: tuple

class Simulator:
    def __init__(self, observation):
        self.obs = observation
        self.w = observation['width']
        self.h = observation['height']
        self.adj = [tuple(y*self.w+x for x,y in ((i%self.w+1,i//self.w),(i%self.w-1,i//self.w),(i%self.w,i//self.w+1),(i%self.w,i//self.w-1)) if 0<=x<self.w and 0<=y<self.h) for i in range(self.w*self.h)]

    def sample(self, rng):
        board = []
        for row in self.obs['board']:
            for c in row:
                t = c['terrain_id']
                if c['visible']:
                    board.append((t,c['owner'],c['army']))
                else:
                    # Hidden mountains may actually be undiscovered castles.
                    if t == 1 and rng.random() < .09: t = 2
                    if t == -2: t = 0
                    board.append((t,-1,rng.randint(40,50) if t==2 else 0))
        # Sample unseen enemy headquarters, never read the real hidden board.
        for p in self.obs['active_players']:
            if not any(t==3 and o==p for t,o,a in board):
                candidates = [i for i,c in enumerate(sum(self.obs['board'],[])) if not c['visible'] and board[i][0]==0 and board[i][1]==-1]
                if candidates:
                    i=rng.choice(candidates); board[i]=(3,p,max(2,self.obs['turn']//3))
        return State(tuple(board),self.obs['turn'],tuple(self.obs['active_players']))

    def actions(self,s,p):
        yield None
        for i,(t,o,a) in enumerate(s.board):
            if o==p and a>1:
                for j in self.adj[i]:
                    if s.board[j][0]!=1:
                        yield (i,j,False)
                        if a>2: yield (i,j,True)

    def step(self,s,moves):
        b=list(s.board); arrivals={}
        for p,m in moves.items():
            if m is None or p not in s.active: continue
            i,j,half=m; t,o,a=s.board[i]
            if o!=p or a<=1 or j not in self.adj[i] or s.board[j][0]==1: continue
            n=a//2 if half else a-1
            b[i]=(t,o,a-n); arrivals.setdefault(j,[]).append((p,n))
        captures=[]
        for j,forces in arrivals.items():
            t,o,a=b[j]; totals={o:a}
            for p,n in forces: totals[p]=totals.get(p,0)+n
            ranked=sorted(totals.items(),key=lambda x:x[1],reverse=True)
            winner,n=ranked[0]; second=ranked[1][1] if len(ranked)>1 else 0
            if n==second: winner=o if totals[o]==n else -1
            b[j]=(t,winner,n-second)
            if t==3 and o>=0 and winner>=0 and winner!=o: captures.append((o,winner))
        active=list(s.active)
        for loser,winner in captures:
            if loser in active:
                active.remove(loser); b=[(t,winner if o==loser else o,a) for t,o,a in b]
        turn=s.turn+1
        b=[(t,o,a+int(o>=0 and t in (2,3))+int(o>=0 and turn%25==0)) for t,o,a in b]
        return State(tuple(b),turn,tuple(active))
