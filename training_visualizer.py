import pygame
import pickle
import os

pygame.init()

WIDTH = 1800
HEIGHT = 1000

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Nathan AI Training Visualizer")

clock = pygame.time.Clock()

FONT = pygame.font.SysFont("consolas", 18)
SMALL = pygame.font.SysFont("consolas", 14)
BIG = pygame.font.SysFont("consolas", 28, bold=True)

latest_generation = -1
replay_data = None
frame_index = 0


def load_replay():

    global replay_data
    global latest_generation
    global frame_index

    if not os.path.exists("live_replay.pkl"):
        return

    try:

        with open("live_replay.pkl", "rb") as f:
            packet = pickle.load(f)

        generation = packet["generation"]

        if generation != latest_generation:

            latest_generation = generation
            replay_data = packet
            frame_index = 0

            print(f"Loaded Generation {generation}")

    except Exception as e:

        print(f"Replay Load Error: {e}")


def get_stats(board):

    stats = {
        0: {"land": 0, "army": 0, "castles": 0},
        1: {"land": 0, "army": 0, "castles": 0},
        2: {"land": 0, "army": 0, "castles": 0},
    }

    for row in board:

        for tile in row:

            owner = tile["owner"]

            if owner not in stats:
                continue

            stats[owner]["land"] += 1
            stats[owner]["army"] += tile["army"]

            if tile["terrain"] == "castle":
                stats[owner]["castles"] += 1

    return stats


def draw_board(state):

    board = state["board"]

    board_width = state["width"]
    board_height = state["height"]

    cell_size = int(
        min(
            950 / board_width,
            950 / board_height
        )
    )

    for row in board:

        for tile in row:

            x = tile["x"]
            y = tile["y"]

            terrain = tile["terrain"]
            owner = tile["owner"]

            color = (80, 80, 80)

            if terrain == "mountain":
                color = (45, 45, 45)

            elif terrain == "castle":
                color = (180, 145, 70)

            elif terrain == "general":
                color = (220, 200, 100)

            if owner == 0:
                color = (70, 140, 255)

            elif owner == 1:
                color = (255, 90, 90)

            elif owner == 2:
                color = (90, 255, 120)

            rect = pygame.Rect(
                x * cell_size,
                y * cell_size,
                cell_size,
                cell_size
            )

            pygame.draw.rect(
                screen,
                color,
                rect
            )

            pygame.draw.rect(
                screen,
                (20, 20, 20),
                rect,
                1
            )

            army = tile["army"]

            if terrain != "mountain" and army > 0:

                text = SMALL.render(
                    str(army),
                    True,
                    (255, 255, 255)
                )

                screen.blit(
                    text,
                    (rect.x + 2, rect.y + 1)
                )


def draw_player_block(
        x,
        y,
        title,
        color,
        stats
):

    pygame.draw.rect(
        screen,
        color,
        (x, y + 4, 18, 18)
    )

    screen.blit(
        FONT.render(
            title,
            True,
            (255, 255, 255)
        ),
        (x + 30, y)
    )

    y += 25

    screen.blit(
        FONT.render(
            f"Land: {stats['land']}",
            True,
            (220, 220, 220)
        ),
        (x + 30, y)
    )

    y += 22

    screen.blit(
        FONT.render(
            f"Army: {stats['army']}",
            True,
            (220, 220, 220)
        ),
        (x + 30, y)
    )

    y += 22

    screen.blit(
        FONT.render(
            f"Castles: {stats['castles']}",
            True,
            (220, 220, 220)
        ),
        (x + 30, y)
    )


running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    load_replay()

    screen.fill((20, 20, 25))

    if replay_data:

        history = replay_data.get("history", [])

        if len(history):

            frame_index = min(
                frame_index,
                len(history) - 1
            )

            state = history[frame_index]

            draw_board(state)

            stats = get_stats(state["board"])

            panel_x = 1050

            screen.blit(
                BIG.render(
                    "TRAINING VIEWER",
                    True,
                    (255, 255, 255)
                ),
                (panel_x, 10)
            )

            info_y = 60

            lines = [

                f"Generation: {replay_data['generation']}",
                f"Turn: {state['turn']}",

                "",

                f"Best Fitness : {replay_data['best']:.2f}",
                f"Avg Fitness  : {replay_data['avg']:.2f}",
                f"Worst Fitness: {replay_data['worst']:.2f}",

                "",

                f"Explore : {replay_data['explore']:.2f}",
                f"Build   : {replay_data['build']:.2f}",
                f"Combat  : {replay_data['combat']:.2f}",
                f"Overlord: {replay_data['overlord']:.2f}",

                "",

                f"Explore Turns: {replay_data['usage'][0]}",
                f"Build Turns  : {replay_data['usage'][1]}",
                f"Combat Turns : {replay_data['usage'][2]}",
            ]

            for line in lines:

                screen.blit(
                    FONT.render(
                        line,
                        True,
                        (255, 255, 255)
                    ),
                    (panel_x, info_y)
                )

                info_y += 22

            draw_player_block(
                panel_x,
                440,
                "Player 0 - NathanAI",
                (70, 140, 255),
                stats[0]
            )

            draw_player_block(
                panel_x,
                560,
                "Player 1 - RandomAI",
                (255, 90, 90),
                stats[1]
            )

            draw_player_block(
                panel_x,
                680,
                "Player 2 - RandomAI",
                (90, 255, 120),
                stats[2]
            )

            if frame_index < len(history) - 1:
                frame_index += 1

    else:

        text = BIG.render(
            "Waiting for live_replay.pkl...",
            True,
            (255, 255, 255)
        )

        screen.blit(text, (50, 50))

    pygame.display.flip()

    clock.tick(6)

pygame.quit()