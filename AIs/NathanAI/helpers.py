# GENERAL HELPERS
def get_enemy_stats(env, player_id):
    enemy_land = []
    enemy_army = []
    for pid in env.get_active_players():
        if pid == player_id:
            continue
        territory, army, _ = env.get_stats(pid)
        enemy_land.append(territory)
        enemy_army.append(army)
    return max(enemy_land, default=0), max(enemy_army, default=0)


def get_visible_enemy_tiles(obs):
    count = 0
    owner_id = obs["player_id"]
    for row in obs["board"]:
        for cell in row:
            if not cell["visible"]:
                continue
            owner = cell["owner"]
            if owner not in (-1, None, owner_id):
                count += 1
    return count


def get_neutral_tiles(obs):
    count = 0
    for row in obs["board"]:
        for cell in row:
            if not cell["visible"]:
                continue
            if cell["owner"] is None:
                count += 1
    return count


def get_visible_cities(obs):
    count = 0
    for row in obs["board"]:
        for cell in row:
            if cell["visible"] and cell["terrain"] == "castle":
                count += 1
    return count


# SNAPSHOTS
def create_snapshot(env, player_id):

    territory, army, castles = env.get_stats(player_id)
    enemy_land = 0
    enemy_army = 0
    for pid in env.get_active_players():
        if pid == player_id:
            continue
        t, a, _ = env.get_stats(pid)
        enemy_land += t
        enemy_army += a

    return {"territory": territory, "army": army, "castles": castles, "enemy_land": enemy_land,
            "enemy_army": enemy_army, "eliminations": len(env.eliminated_players),
            "frontline_army": frontline_score(env, player_id)}


# CANDIDATE ACTION FEATURES
def get_action_features(env, player_id, action):

    if action.is_pass:
        return [0] * 9

    source = env.tile(action.source)
    target = env.tile(action.destination)

    if target.owner == player_id:
        owner_type = 0

    elif target.owner == -1:
        owner_type = 1

    else:
        owner_type = 2

    target_army = target.army
    army_advantage = (source.army - target_army)
    capture_possible = int(source.army > target_army + 1)

    friendly_neighbors = 0
    enemy_neighbors = 0

    for n in env.neighbors(action.destination):
        n_tile = env.tile(n)
        if n_tile.owner == player_id:
            friendly_neighbors += 1

        elif n_tile.owner not in (-1, player_id):
            enemy_neighbors += 1

    return [float(source.army), float(target_army), float(owner_type), float(int(target.terrain)),
            float(capture_possible), float(army_advantage), float(friendly_neighbors), float(enemy_neighbors),
            float(int(action.half))]


def build_candidate_action_block(env, player_id, actions, limit=50):
    features = []
    for action in actions:
        features.extend(get_action_features(env, player_id, action))

    while len(features) < limit * 9:
        features.extend([0] * 9)

    return features


def get_candidate_actions(env, player_id, limit=50):

    actions = env.get_legal_actions(player_id, include_pass=False)

    def score(action):
        source = env.tile(action.source)
        target = env.tile(action.destination)
        value = 0

        # Bigger stacks are more meaningful
        value += source.army

        # Expand aggressively
        if target.owner is None:
            value += 100

        # Capture city
        if target.terrain == "castle":
            value += 300

        # Attack enemy
        if target.owner not in (-1, None, player_id):
            value += 200

        # Winnable attack
        if source.army > target.army + 1:
            value += 100
        value += (source.army - target.army)
        return value

    actions.sort(key=score, reverse=True)

    return actions[:limit]


def get_frontline_tiles(env, player_id):
    count = 0
    for y in range(env.height):
        for x in range(env.width):
            tile = env.tile((x, y))

            if tile.owner != player_id:
                continue

            for n in env.neighbors((x, y)):
                neighbor = env.tile(n)

                if neighbor.owner not in (-1, player_id):
                    count += 1
                    break
    return count


def get_largest_stack(env, player_id):
    largest = 0

    for row in env.board:
        for tile in row:
            if tile.owner == player_id:
                largest = max(largest, tile.army)
    return largest


def enemy_seen(obs):
    return int(get_visible_enemy_tiles(obs) > 0)


def frontline_score(env, player_id):

    score = 0

    for y in range(env.height):
        for x in range(env.width):

            pos = (x, y)
            tile = env.tile(pos)

            if tile.owner != player_id:
                continue

            touching_frontier = False

            for n in env.neighbors(pos):

                n_tile = env.tile(n)

                if n_tile.owner != player_id:

                    touching_frontier = True
                    break

            if touching_frontier:
                score += tile.army

    return score

def nearest_enemy_info(env, player_id):
    general = env.generals[player_id]

    best_distance = 999
    best_dx = 0
    best_dy = 0

    for y in range(env.height):
        for x in range(env.width):

            tile = env.tile((x, y))

            if tile.owner in (-1, player_id):
                continue

            dx = x - general[0]
            dy = y - general[1]

            dist = abs(dx) + abs(dy)

            if dist < best_distance:
                best_distance = dist
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy, best_distance
