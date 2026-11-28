"""Hand-designed strategic evaluation; no learned weights."""
def evaluate(state, player, sim):
    if player not in state.active: return -1000.0
    if len(state.active)==1: return 1000.0
    scores={p:0.0 for p in state.active}
    for i,(t,o,a) in enumerate(state.board):
        if o not in scores: continue
        scores[o]+=3+0.22*a+ (15 if t==2 else 0)
        if t==3:
            threat=max((state.board[j][2]-1 for j in sim.adj[i] if state.board[j][1]>=0 and state.board[j][1]!=o),default=0)
            scores[o]-=12*max(0,threat-a+1)
        if any(state.board[j][1]!=o and state.board[j][0]!=1 for j in sim.adj[i]):
            scores[o]+=0.12*a
    return scores[player]-.55*max((v for p,v in scores.items() if p!=player),default=0)

def move_score(s,p,m,sim):
    if m is None: return -.5
    i,j,half=m; t,o,a=s.board[i]; dt,do,da=s.board[j]
    n=a//2 if half else a-1
    if dt==1: return -100
    if do!=p:
        score=5+min(n-da,20)*.12 if n>da else -8-da*.1
        if n>da and dt==2: score+=12
        if n>da and dt==3: score+=80
    else:
        score=.1*n if any(s.board[k][1]!=p and s.board[k][0]!=1 for k in sim.adj[j]) else -.8
    if t==3:
        threat=max((s.board[k][2]-1 for k in sim.adj[i] if s.board[k][1]>=0 and s.board[k][1]!=p),default=0)
        score-=15*max(0,threat-(a-n)+1)
    return score
