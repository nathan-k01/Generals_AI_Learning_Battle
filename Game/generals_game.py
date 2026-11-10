"""
**Warning**
This file is primarily AI generated with a few human modifications. For the purposes of this research project
the manual building of the game environment was not as important as the building of the player AIs themselves
**Warning**

Generals Arena: Three-Player AI Environment

Install:
    python -m pip install pygame

Run:
    python generals_arena.py

Run with a reproducible map:
    python generals_arena.py --seed 12345

Important engine methods:
    environment.get_observation(player_id)
    environment.get_full_state()
    environment.get_legal_actions(player_id)
    environment.submit_action(player_id, action)
    environment.get_pending_players()
    environment.is_waiting_for_player(player_id)
    environment.reset(seed=None)

The engine waits until players 0, 1, and 2 have each submitted one
action. It then resolves the actions simultaneously and advances the turn.

The Pygame interface is a local debugging interface. Press 1, 2, or 3
to switch between players while submitting actions.
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Optional

import pygame


# ============================================================
# CONSTANTS
# ============================================================

PLAYER_IDS = (0, 1, 2)
NEUTRAL = -1
UNKNOWN = -2

DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


# ============================================================
# DATA TYPES
# ============================================================

class Terrain(IntEnum):
    PLAIN = 0
    MOUNTAIN = 1
    CASTLE = 2
    GENERAL = 3


@dataclass(frozen=True)
class Action:
    """
    Represents one player's action for a turn.

    A pass action has no source or destination.

    Examples:
        Action.pass_action()

        Action(
            source=(4, 7),
            destination=(5, 7),
            half=False,
        )
    """

    source: Optional[tuple[int, int]] = None
    destination: Optional[tuple[int, int]] = None
    half: bool = False

    @property
    def is_pass(self) -> bool:
        return (
            self.source is None
            or self.destination is None
        )

    @classmethod
    def pass_action(cls) -> "Action":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "half": self.half,
            "is_pass": self.is_pass,
        }

    @classmethod
    def from_dict(
        cls,
        action_data: dict[str, Any],
    ) -> "Action":
        if action_data.get("type") == "pass":
            return cls.pass_action()

        source = action_data.get("source")
        destination = action_data.get("destination")

        if source is not None:
            source = tuple(source)

        if destination is not None:
            destination = tuple(destination)

        return cls(
            source=source,
            destination=destination,
            half=bool(action_data.get("half", False)),
        )


@dataclass
class Tile:
    terrain: Terrain = Terrain.PLAIN
    owner: int = NEUTRAL
    army: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "terrain": self.terrain.name.lower(),
            "terrain_id": int(self.terrain),
            "owner": self.owner,
            "army": self.army,
        }


@dataclass
class SubmittedAction:
    player_id: int
    action: Action


# ============================================================
# GAME ENVIRONMENT
# ============================================================

class GeneralsEnvironment:
    """
    Three-player simultaneous-turn Generals environment.

    External agents interact with the environment using:

        get_observation(player_id)
        get_legal_actions(player_id)
        submit_action(player_id, action)

    A turn resolves automatically when all active players have
    submitted an action.

    Submitted actions are cleared immediately after the turn resolves.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        fog_of_war: bool = True,
        max_turns: int = 1200,
    ):
        self.fog_of_war = fog_of_war
        self.max_turns = max_turns

        self.seed = (
            seed
            if seed is not None
            else random.SystemRandom().randrange(1 << 63)
        )

        self.rng = random.Random(self.seed)

        self.width = self.rng.randint(18, 21)
        self.height = self.rng.randint(18, 21)

        self.turn = 0
        self.game_over = False
        self.winner: Optional[int] = None
        self.draw = False

        self.board: list[list[Tile]] = []
        self.generals: dict[int, tuple[int, int]] = {}

        self.eliminated_players: set[int] = set()

        # This dictionary acts as the current turn's input fields.
        self.pending_actions: dict[int, Action] = {}

        # Stores terrain that each player has previously seen.
        self.known_terrain: dict[
            int,
            list[list[int]],
        ] = {
            player_id: [
                [UNKNOWN for _ in range(self.width)]
                for _ in range(self.height)
            ]
            for player_id in PLAYER_IDS
        }

        self.last_turn_result: dict[str, Any] = {}

        self._generate_map()
        self._initialize_public_terrain_knowledge()
        self._update_all_player_knowledge()

    # --------------------------------------------------------
    # PUBLIC ENVIRONMENT API
    # --------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> dict[int, dict[str, Any]]:
        """
        Reset the environment and return initial observations.
        """

        self.__init__(
            seed=seed,
            fog_of_war=self.fog_of_war,
            max_turns=self.max_turns,
        )

        return {
            player_id: self.get_observation(player_id)
            for player_id in PLAYER_IDS
        }

    def submit_action(
        self,
        player_id: int,
        action: Action | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Submit an action for one player.

        The turn resolves automatically after every active player
        has submitted exactly one action.

        Invalid actions are accepted but treated as passes when the
        turn resolves.

        Returns a status dictionary describing whether the engine is
        still waiting or whether the turn was resolved.
        """

        self._validate_player_id(player_id)

        if self.game_over:
            return {
                "accepted": False,
                "turn_resolved": False,
                "reason": "game_over",
                "turn": self.turn,
                "winner": self.winner,
                "draw": self.draw,
            }

        if player_id in self.eliminated_players:
            return {
                "accepted": False,
                "turn_resolved": False,
                "reason": "player_eliminated",
                "player_id": player_id,
                "turn": self.turn,
            }

        if player_id in self.pending_actions:
            return {
                "accepted": False,
                "turn_resolved": False,
                "reason": "action_already_submitted",
                "player_id": player_id,
                "turn": self.turn,
                "waiting_for": self.get_pending_players(),
            }

        if isinstance(action, dict):
            action = Action.from_dict(action)

        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action or action dictionary"
            )

        self.pending_actions[player_id] = action

        waiting_for = self.get_pending_players()

        if waiting_for:
            return {
                "accepted": True,
                "turn_resolved": False,
                "player_id": player_id,
                "turn": self.turn,
                "waiting_for": waiting_for,
                "submitted_players": sorted(
                    self.pending_actions.keys()
                ),
            }

        turn_result = self._resolve_pending_turn()

        return {
            "accepted": True,
            "turn_resolved": True,
            "turn": self.turn,
            "waiting_for": self.get_pending_players(),
            "result": turn_result,
            "winner": self.winner,
            "draw": self.draw,
            "game_over": self.game_over,
        }

    def replace_submitted_action(
        self,
        player_id: int,
        action: Action | dict[str, Any],
    ) -> bool:
        """
        Replace an action before the final required player submits.

        This is optional convenience behavior for a controller or UI.
        It cannot replace an action after the turn has resolved because
        pending actions are reset after resolution.
        """

        self._validate_player_id(player_id)

        if player_id not in self.pending_actions:
            return False

        if isinstance(action, dict):
            action = Action.from_dict(action)

        self.pending_actions[player_id] = action
        return True

    def withdraw_action(self, player_id: int) -> bool:
        """
        Remove a submitted action before the turn resolves.
        """

        self._validate_player_id(player_id)

        if player_id not in self.pending_actions:
            return False

        del self.pending_actions[player_id]
        return True

    def is_waiting_for_player(
        self,
        player_id: int,
    ) -> bool:
        """
        Return True when the player still needs to submit this turn.
        """

        self._validate_player_id(player_id)

        return (
            not self.game_over
            and player_id not in self.eliminated_players
            and player_id not in self.pending_actions
        )

    def has_player_submitted(
        self,
        player_id: int,
    ) -> bool:
        self._validate_player_id(player_id)
        return player_id in self.pending_actions

    def get_pending_players(self) -> list[int]:
        """
        Return active players who have not submitted this turn.
        """

        if self.game_over:
            return []

        return [
            player_id
            for player_id in self.get_active_players()
            if player_id not in self.pending_actions
        ]

    def get_active_players(self) -> list[int]:
        return [
            player_id
            for player_id in PLAYER_IDS
            if player_id not in self.eliminated_players
        ]

    def get_submitted_actions(
        self,
    ) -> dict[int, dict[str, Any]]:
        """
        Return submitted actions.

        This should normally be used only by an administrator or
        debugging interface. Agents should not be told another
        player's pending action.
        """

        return {
            player_id: action.to_dict()
            for player_id, action
            in self.pending_actions.items()
        }

    # ============================================================
    # PUBLIC TERRAIN AND FOG-OF-WAR SETUP
    # ============================================================

    def _initialize_public_terrain_knowledge(self):
        """
        Give every player global terrain knowledge for mountains and cities.

        Rules:
        - Real mountains are visible to every player immediately.
        - Undiscovered cities are displayed as mountains.
        - Plains and generals remain unknown until normal vision reveals them.
        - Once a player sees a city, that player's known terrain is updated
          from MOUNTAIN to CASTLE and remains remembered as a city.
        """

        for player_id in PLAYER_IDS:
            for y in range(self.height):
                for x in range(self.width):
                    tile = self.tile((x, y))

                    if tile.terrain == Terrain.MOUNTAIN:
                        self.known_terrain[player_id][y][x] = int(
                            Terrain.MOUNTAIN
                        )

                    elif tile.terrain == Terrain.CASTLE:
                        # Cities appear to be mountains until the player
                        # obtains normal fog-of-war vision of the tile.
                        self.known_terrain[player_id][y][x] = int(
                            Terrain.MOUNTAIN
                        )

                    else:
                        self.known_terrain[player_id][y][x] = UNKNOWN
    def get_observation(
        self,
        player_id: int,
    ) -> dict[str, Any]:
        """
        Return the fog-of-war observation for one player.

        Hidden cells do not expose owner or army information.

        Previously observed terrain remains known, but ownership and
        army values become hidden when the cell leaves vision.
        """

        self._validate_player_id(player_id)

        visible = self.get_visible_cells(player_id)

        cells = []

        for y in range(self.height):
            row = []

            for x in range(self.width):
                position = (x, y)
                tile = self.tile(position)
                is_visible = position in visible

                known_terrain_id = (
                    self.known_terrain[player_id][y][x]
                )

                if not self.fog_of_war or is_visible:
                    row.append(
                        {
                            "x": x,
                            "y": y,
                            "visible": True,
                            "terrain": tile.terrain.name.lower(),
                            "terrain_id": int(tile.terrain),
                            "owner": tile.owner,
                            "army": tile.army,
                        }
                    )

                elif known_terrain_id != UNKNOWN:
                    known_terrain = Terrain(
                        known_terrain_id
                    )

                    row.append(
                        {
                            "x": x,
                            "y": y,
                            "visible": False,
                            "terrain": (
                                known_terrain.name.lower()
                            ),
                            "terrain_id": int(
                                known_terrain
                            ),
                            "owner": UNKNOWN,
                            "army": UNKNOWN,
                        }
                    )

                else:
                    row.append(
                        {
                            "x": x,
                            "y": y,
                            "visible": False,
                            "terrain": "unknown",
                            "terrain_id": UNKNOWN,
                            "owner": UNKNOWN,
                            "army": UNKNOWN,
                        }
                    )

            cells.append(row)

        territory, total_army, castles = self.get_stats(
            player_id
        )

        return {
            "player_id": player_id,
            "turn": self.turn,
            "width": self.width,
            "height": self.height,
            "max_turns": self.max_turns,
            "game_over": self.game_over,
            "winner": self.winner,
            "draw": self.draw,
            "eliminated": (
                player_id in self.eliminated_players
            ),
            "active_players": self.get_active_players(),
            "action_submitted": (
                player_id in self.pending_actions
            ),
            "waiting_for_input": self.is_waiting_for_player(
                player_id
            ),
            "stats": {
                "territory": territory,
                "total_army": total_army,
                "castles": castles,
            },
            "board": cells,
        }

    def get_full_state(self) -> dict[str, Any]:
        """
        Return perfect-information state.

        This is intended for referees, logging, replay generation,
        testing, or centralized training.

        Do not provide this result to a fog-of-war agent during normal
        competition play.
        """

        board_data = []

        for y in range(self.height):
            row = []

            for x in range(self.width):
                tile_data = self.tile((x, y)).to_dict()
                tile_data["x"] = x
                tile_data["y"] = y
                row.append(tile_data)

            board_data.append(row)

        return {
            "seed": self.seed,
            "turn": self.turn,
            "width": self.width,
            "height": self.height,
            "max_turns": self.max_turns,
            "game_over": self.game_over,
            "winner": self.winner,
            "draw": self.draw,
            "active_players": self.get_active_players(),
            "eliminated_players": sorted(
                self.eliminated_players
            ),
            "generals": copy.deepcopy(self.generals),
            "pending_players": self.get_pending_players(),
            "submitted_players": sorted(
                self.pending_actions.keys()
            ),
            "board": board_data,
            "last_turn_result": copy.deepcopy(
                self.last_turn_result
            ),
        }

    def get_legal_actions(
        self,
        player_id: int,
        include_pass: bool = True,
    ) -> list[Action]:
        """
        Return actions that are legal from the current true state.

        This method only generates moves from cells owned byl hidden destination army
        counts.

        Mountain destinations are omitted only when their terrain is
        currently known to that player. An attempted move into an
        unknown mountain may still be submitted and will become a pass.
        """

        self._validate_player_id(player_id)

        if (
            self.game_over
            or player_id in self.eliminated_players
        ):
            return []

        legal_actions = []

        if include_pass:
            legal_actions.append(Action.pass_action())

        for y in range(self.height):
            for x in range(self.width):
                source = (x, y)
                source_tile = self.tile(source)

                if source_tile.owner != player_id:
                    continue

                if source_tile.army <= 1:
                    continue

                for destination in self.neighbors(source):
                    destination_x, destination_y = (
                        destination
                    )

                    known_terrain = (
                        self.known_terrain[player_id]
                        [destination_y]
                        [destination_x]
                    )

                    if known_terrain == int(
                        Terrain.MOUNTAIN
                    ):
                        continue

                    legal_actions.append(
                        Action(
                            source=source,
                            destination=destination,
                            half=False,
                        )
                    )

                    legal_actions.append(
                        Action(
                            source=source,
                            destination=destination,
                            half=True,
                        )
                    )

        return legal_actions

    def get_visible_cells(
        self,
        player_id: int,
    ) -> set[tuple[int, int]]:
        """
        A player sees every owned tile and all orthogonally and
        diagonally adjacent tiles.

        Generals.io-style vision normally includes the eight cells
        surrounding owned territory.
        """

        self._validate_player_id(player_id)

        visible: set[tuple[int, int]] = set()

        for y in range(self.height):
            for x in range(self.width):
                position = (x, y)

                if self.tile(position).owner != player_id:
                    continue

                visible.add(position)

                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        visible_position = (
                            x + offset_x,
                            y + offset_y,
                        )

                        if self.in_bounds(
                            visible_position
                        ):
                            visible.add(
                                visible_position
                            )

        return visible

    def get_stats(
        self,
        player_id: int,
    ) -> tuple[int, int, int]:
        self._validate_player_id(player_id)

        territory = 0
        total_army = 0
        castles = 0

        for row in self.board:
            for tile in row:
                if tile.owner != player_id:
                    continue

                territory += 1
                total_army += tile.army

                if tile.terrain == Terrain.CASTLE:
                    castles += 1

        return territory, total_army, castles

    # --------------------------------------------------------
    # TURN RESOLUTION
    # --------------------------------------------------------

    def _resolve_pending_turn(self) -> dict[str, Any]:
        """
        Resolve the turn after all active players have submitted.

        Processing order:
        1. Validate submitted actions.
        2. Remove all outgoing troops.
        3. Combine forces arriving at each destination.
        4. Resolve destination battles.
        5. Process general captures and territory transfers.
        6. Apply growth.
        7. Update fog-of-war knowledge.
        8. Clear pending actions.
        """

        active_players_at_start = self.get_active_players()

        validated_actions = {
            player_id: self._validate_action(
                player_id,
                self.pending_actions.get(
                    player_id,
                    Action.pass_action(),
                ),
            )
            for player_id in active_players_at_start
        }

        submitted_snapshot = {
            player_id: self.pending_actions[
                player_id
            ].to_dict()
            for player_id in self.pending_actions
        }

        arrivals: dict[
            tuple[int, int],
            list[tuple[int, int]],
        ] = {}

        valid_move_results = {}

        for player_id, validated in (
            validated_actions.items()
        ):
            if validated is None:
                valid_move_results[player_id] = {
                    "valid": False,
                    "treated_as": "pass",
                }
                continue

            source, destination, sent_army = validated

            self.tile(source).army -= sent_army

            arrivals.setdefault(
                destination,
                [],
            ).append(
                (player_id, sent_army)
            )

            valid_move_results[player_id] = {
                "valid": True,
                "source": source,
                "destination": destination,
                "sent_army": sent_army,
            }

        captured_generals: list[
            tuple[int, int]
        ] = []

        battle_results = []

        for destination, arriving_forces in (
            arrivals.items()
        ):
            destination_tile = self.tile(destination)

            old_owner = destination_tile.owner
            old_army = destination_tile.army

            force_totals = {
                NEUTRAL: 0,
                0: 0,
                1: 0,
                2: 0,
            }

            force_totals[old_owner] += old_army

            for owner, amount in arriving_forces:
                force_totals[owner] += amount

            positive_forces = [
                (owner, force)
                for owner, force
                in force_totals.items()
                if force > 0
            ]

            positive_forces.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            if not positive_forces:
                destination_tile.owner = old_owner
                destination_tile.army = 0

            else:
                strongest_force = positive_forces[0][1]

                tied_for_first = [
                    owner
                    for owner, force
                    in positive_forces
                    if force == strongest_force
                ]

                if len(tied_for_first) > 1:
                    # Exact ties favor the existing defender if the
                    # existing defender is part of the tie.
                    if old_owner in tied_for_first:
                        destination_tile.owner = old_owner
                        destination_tile.army = 0
                    else:
                        destination_tile.owner = NEUTRAL
                        destination_tile.army = 0

                else:
                    strongest_owner = (
                        positive_forces[0][0]
                    )

                    second_strongest_force = (
                        positive_forces[1][1]
                        if len(positive_forces) > 1
                        else 0
                    )

                    destination_tile.owner = (
                        strongest_owner
                    )

                    destination_tile.army = (
                        strongest_force
                        - second_strongest_force
                    )

            if (
                destination_tile.terrain
                == Terrain.GENERAL
                and old_owner in PLAYER_IDS
                and destination_tile.owner != old_owner
                and destination_tile.owner in PLAYER_IDS
            ):
                captured_generals.append(
                    (
                        old_owner,
                        destination_tile.owner,
                    )
                )

            battle_results.append(
                {
                    "destination": destination,
                    "previous_owner": old_owner,
                    "previous_army": old_army,
                    "arrivals": [
                        {
                            "player_id": owner,
                            "army": amount,
                        }
                        for owner, amount
                        in arriving_forces
                    ],
                    "new_owner": destination_tile.owner,
                    "new_army": destination_tile.army,
                }
            )

        elimination_results = []

        for defeated_player, capturing_player in (
            captured_generals
        ):
            if defeated_player in self.eliminated_players:
                continue

            transferred_tiles = 0

            for row in self.board:
                for tile in row:
                    if tile.owner == defeated_player:
                        tile.owner = capturing_player
                        transferred_tiles += 1

            self.eliminated_players.add(
                defeated_player
            )

            elimination_results.append(
                {
                    "defeated_player": defeated_player,
                    "capturing_player": capturing_player,
                    "transferred_tiles": transferred_tiles,
                }
            )

        self.turn += 1

        self._apply_growth()
        self._update_all_player_knowledge()
        self._update_game_end_state()

        # Reset all input fields for the next turn.
        self.pending_actions.clear()

        self.last_turn_result = {
            "resolved_turn": self.turn,
            "submitted_actions": submitted_snapshot,
            "validated_actions": valid_move_results,
            "battles": battle_results,
            "eliminations": elimination_results,
            "active_players": self.get_active_players(),
            "winner": self.winner,
            "draw": self.draw,
            "game_over": self.game_over,
        }

        return copy.deepcopy(self.last_turn_result)

    def _validate_action(
        self,
        player_id: int,
        action: Action,
    ) -> Optional[
        tuple[
            tuple[int, int],
            tuple[int, int],
            int,
        ]
    ]:
        """
        Return validated move information or None.

        Invalid moves silently become passes.
        """

        if action.is_pass:
            return None

        if action.source is None:
            return None

        if action.destination is None:
            return None

        if not self.in_bounds(action.source):
            return None

        if not self.in_bounds(action.destination):
            return None

        source_x, source_y = action.source
        destination_x, destination_y = (
            action.destination
        )

        manhattan_distance = (
            abs(source_x - destination_x)
            + abs(source_y - destination_y)
        )

        if manhattan_distance != 1:
            return None

        source_tile = self.tile(action.source)
        destination_tile = self.tile(
            action.destination
        )

        if source_tile.owner != player_id:
            return None

        if source_tile.army <= 1:
            return None

        if (
            destination_tile.terrain
            == Terrain.MOUNTAIN
        ):
            return None

        if action.half:
            sent_army = source_tile.army // 2
        else:
            sent_army = source_tile.army - 1

        if sent_army <= 0:
            return None

        return (
            action.source,
            action.destination,
            sent_army,
        )

    def _apply_growth(self):
        for row in self.board:
            for tile in row:
                if tile.terrain in (
                    Terrain.GENERAL,
                    Terrain.CASTLE,
                ):
                    if tile.owner in PLAYER_IDS:
                        tile.army += 1

                    elif (
                        tile.terrain == Terrain.CASTLE
                        and tile.army < 40
                    ):
                        tile.army += 1

                if (
                    self.turn % 25 == 0
                    and tile.owner in PLAYER_IDS
                ):
                    tile.army += 1

    def _update_game_end_state(self):
        active_players = self.get_active_players()

        if len(active_players) == 1:
            self.game_over = True
            self.winner = active_players[0]
            self.draw = False
            return

        if self.turn >= self.max_turns:
            self.game_over = True
            self.winner = None
            self.draw = True

    # --------------------------------------------------------
    # MAP GENERATION
    # --------------------------------------------------------

    def _generate_map(self):
        for _ in range(4000):
            self.board = [
                [Tile() for _ in range(self.width)]
                for _ in range(self.height)
            ]

            all_positions = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
            ]

            mountain_density = self.rng.uniform(
                0.24,
                0.26,
            )

            mountain_count = round(
                len(all_positions)
                * mountain_density
            )

            for position in self.rng.sample(
                all_positions,
                mountain_count,
            ):
                self.tile(position).terrain = (
                    Terrain.MOUNTAIN
                )

            mountain_positions = [
                position
                for position in all_positions
                if (
                    self.tile(position).terrain
                    == Terrain.MOUNTAIN
                )
            ]

            castle_count = min(
                self.rng.randint(9, 11),
                len(mountain_positions),
            )

            for position in self.rng.sample(
                mountain_positions,
                castle_count,
            ):
                tile = self.tile(position)
                tile.terrain = Terrain.CASTLE
                tile.army = self.rng.randint(40, 50)

            passable_positions = [
                position
                for position in all_positions
                if (
                    self.tile(position).terrain
                    != Terrain.MOUNTAIN
                )
            ]

            components = self._find_components(
                passable_positions
            )

            if not components:
                continue

            largest_component = max(
                components,
                key=len,
            )

            general_positions = (
                self._find_three_general_positions(
                    largest_component
                )
            )

            if general_positions is None:
                continue

            self.generals = {
                player_id: general_positions[player_id]
                for player_id in PLAYER_IDS
            }

            for player_id, position in (
                self.generals.items()
            ):
                tile = self.tile(position)
                tile.terrain = Terrain.GENERAL
                tile.owner = player_id
                tile.army = 1

            return

        raise RuntimeError(
            "Could not generate a valid three-player map."
        )

    def _find_components(
        self,
        passable_positions: list[
            tuple[int, int]
        ],
    ) -> list[set[tuple[int, int]]]:
        remaining = set(passable_positions)
        components = []

        while remaining:
            start = next(iter(remaining))

            component = {start}
            queue = deque([start])

            remaining.remove(start)

            while queue:
                current = queue.popleft()

                for neighbor in self.neighbors(current):
                    if neighbor not in remaining:
                        continue

                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

            components.append(component)

        return components

    def _find_three_general_positions(
        self,
        component: set[tuple[int, int]],
    ) -> Optional[list[tuple[int, int]]]:
        """
        Find three plain tiles where every pair is at least 17 BFS
        steps apart.
        """

        candidates = [
            position
            for position in component
            if (
                self.tile(position).terrain
                == Terrain.PLAIN
            )
        ]

        self.rng.shuffle(candidates)

        allowed = set(component)

        distance_cache: dict[
            tuple[int, int],
            dict[tuple[int, int], int],
        ] = {}

        search_candidates = candidates[:200]

        for first in search_candidates:
            distance_cache[first] = self._distances(
                first,
                allowed,
            )

            second_candidates = [
                position
                for position in search_candidates
                if (
                    distance_cache[first].get(
                        position,
                        -1,
                    )
                    >= 17
                )
            ]

            self.rng.shuffle(second_candidates)

            for second in second_candidates[:100]:
                if second not in distance_cache:
                    distance_cache[second] = (
                        self._distances(
                            second,
                            allowed,
                        )
                    )

                third_candidates = [
                    position
                    for position in search_candidates
                    if (
                        position != first
                        and position != second
                        and (
                            distance_cache[first].get(
                                position,
                                -1,
                            )
                            >= 17
                        )
                        and (
                            distance_cache[second].get(
                                position,
                                -1,
                            )
                            >= 17
                        )
                    )
                ]

                if third_candidates:
                    third = self.rng.choice(
                        third_candidates
                    )

                    positions = [
                        first,
                        second,
                        third,
                    ]

                    self.rng.shuffle(positions)
                    return positions

        return None

    def _distances(
        self,
        start: tuple[int, int],
        allowed: set[tuple[int, int]],
    ) -> dict[tuple[int, int], int]:
        distances = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor in self.neighbors(current):
                if neighbor not in allowed:
                    continue

                if neighbor in distances:
                    continue

                distances[neighbor] = (
                    distances[current] + 1
                )

                queue.append(neighbor)

        return distances

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def tile(
        self,
        position: tuple[int, int],
    ) -> Tile:
        x, y = position
        return self.board[y][x]

    def in_bounds(
        self,
        position: tuple[int, int],
    ) -> bool:
        x, y = position

        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    def neighbors(
        self,
        position: tuple[int, int],
    ):
        x, y = position

        for delta_x, delta_y in DIRECTIONS:
            neighbor = (
                x + delta_x,
                y + delta_y,
            )

            if self.in_bounds(neighbor):
                yield neighbor

    def _validate_player_id(
        self,
        player_id: int,
    ):
        if player_id not in PLAYER_IDS:
            raise ValueError(
                f"player_id must be one of {PLAYER_IDS}"
            )

    # ============================================================
    # PUBLIC TERRAIN AND FOG-OF-WAR SETUP
    # ============================================================

    def _initialize_public_terrain_knowledge(self):
        """
        Give every player global terrain knowledge for mountains and cities.

        Rules:
        - Real mountains are visible to every player immediately.
        - Undiscovered cities are displayed as mountains.
        - Plains and generals remain unknown until normal vision reveals them.
        - Once a player sees a city, that player's known terrain is updated
          from MOUNTAIN to CASTLE and remains remembered as a city.
        """

        for player_id in PLAYER_IDS:
            for y in range(self.height):
                for x in range(self.width):
                    tile = self.tile((x, y))

                    if tile.terrain == Terrain.MOUNTAIN:
                        self.known_terrain[player_id][y][x] = int(
                            Terrain.MOUNTAIN
                        )

                    elif tile.terrain == Terrain.CASTLE:
                        # Cities appear to be mountains until the player
                        # obtains normal fog-of-war vision of the tile.
                        self.known_terrain[player_id][y][x] = int(
                            Terrain.MOUNTAIN
                        )

                    else:
                        self.known_terrain[player_id][y][x] = UNKNOWN

    def _update_all_player_knowledge(self):
        for player_id in PLAYER_IDS:
            visible = self.get_visible_cells(
                player_id
            )

            for x, y in visible:
                self.known_terrain[player_id][y][x] = (
                    int(self.tile((x, y)).terrain)
                )


# ============================================================
# PYGAME DEBUGGING INTERFACE
# ============================================================

class GeneralsApp:
    BACKGROUND = (20, 24, 31)
    GRID_COLOR = (48, 54, 66)

    PLAIN_COLOR = (91, 103, 116)
    MOUNTAIN_COLOR = (52, 57, 65)
    CASTLE_COLOR = (176, 144, 78)
    UNKNOWN_COLOR = (14, 17, 22)

    PLAYER_COLORS = {
        0: (63, 139, 255),
        1: (238, 80, 88),
        2: (66, 196, 124),
    }

    PLAYER_NAMES = {
        0: "PLAYER 1",
        1: "PLAYER 2",
        2: "PLAYER 3",
    }

    TEXT_COLOR = (235, 238, 244)
    MUTED_TEXT_COLOR = (165, 173, 187)
    SELECTION_COLOR = (255, 221, 86)
    QUEUED_COLOR = (255, 255, 255)

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        pygame.init()
        pygame.display.set_caption(
            "Generals Arena: Three-Player Environment"
        )

        self.screen = pygame.display.set_mode(
            (1250, 840),
            pygame.RESIZABLE,
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            18,
        )

        self.small_font = pygame.font.SysFont(
            "consolas",
            14,
        )

        self.large_font = pygame.font.SysFont(
            "consolas",
            29,
            bold=True,
        )

        self.environment = GeneralsEnvironment(
            seed=seed
        )

        self.current_player = 0
        self.selected: Optional[
            tuple[int, int]
        ] = None

        self.queued_actions: dict[int, Action] = {}

        self.use_half = False
        self.show_fog = True
        self.spectator_mode = False

        self.message = (
            "Player 1: select an owned tile."
        )

    def restart(self):
        self.environment.reset()
        self.current_player = 0
        self.selected = None
        self.queued_actions.clear()
        self.use_half = False

        self.message = (
            "New match. Player 1 may submit."
        )

    def switch_player(
        self,
        player_id: int,
    ):
        if player_id not in PLAYER_IDS:
            return

        self.current_player = player_id
        self.selected = None

        if self.environment.has_player_submitted(
            player_id
        ):
            self.message = (
                f"{self.PLAYER_NAMES[player_id]} "
                "has already submitted."
            )
        else:
            self.message = (
                f"{self.PLAYER_NAMES[player_id]} "
                "is now active."
            )

    def calculate_layout(self):
        screen_width, screen_height = (
            self.screen.get_size()
        )

        panel_width = 345

        available_width = max(
            100,
            screen_width - panel_width - 30,
        )

        available_height = max(
            100,
            screen_height - 65,
        )

        cell_size = max(
            12,
            min(
                38,
                available_width
                // self.environment.width,
                available_height
                // self.environment.height,
            ),
        )

        board_width = (
            cell_size * self.environment.width
        )

        board_height = (
            cell_size * self.environment.height
        )

        offset_x = 15 + max(
            0,
            (available_width - board_width) // 2,
        )

        offset_y = 15 + max(
            0,
            (available_height - board_height) // 2,
        )

        return (
            offset_x,
            offset_y,
            cell_size,
            panel_width,
        )

    def point_to_cell(
        self,
        mouse_position: tuple[int, int],
    ) -> Optional[tuple[int, int]]:
        offset_x, offset_y, cell_size, _ = (
            self.calculate_layout()
        )

        x = (
            mouse_position[0] - offset_x
        ) // cell_size

        y = (
            mouse_position[1] - offset_y
        ) // cell_size

        position = (int(x), int(y))

        if self.environment.in_bounds(position):
            return position

        return None

    def handle_click(self, position: tuple[int, int]):
        """Queue moves to either friendly or non-friendly adjacent tiles."""
        if self.environment.game_over:
            return

        if self.environment.has_player_submitted(
                self.current_player
        ):
            self.message = (
                "This player already submitted an action."
            )
            return

        tile = self.environment.tile(position)

        # If a source is already selected, first interpret any
        # adjacent click as a destination. This includes another
        # tile owned by the same player.
        if self.selected is not None:
            source_x, source_y = self.selected
            target_x, target_y = position

            distance = (
                    abs(source_x - target_x)
                    + abs(source_y - target_y)
            )

            if distance == 1:
                self.queued_actions[
                    self.current_player
                ] = Action(
                    source=self.selected,
                    destination=position,
                    half=self.use_half,
                )

                self.message = (
                    f"{self.PLAYER_NAMES[self.current_player]} "
                    "move queued. Press Enter to submit."
                )
                return

        # Clicking a non-adjacent owned tile changes the source.
        if tile.owner == self.current_player:
            self.selected = position

            self.queued_actions.pop(
                self.current_player,
                None,
            )

            self.message = (
                f"Selected {position} with "
                f"{tile.army} units."
            )
            return

        if self.selected is None:
            self.message = (
                "Select one of the current player's "
                "owned tiles first."
            )
        else:
            self.message = (
                "The destination must be "
                "orthogonally adjacent."
            )

        tile = self.environment.tile(position)

        if tile.owner == self.current_player:
            self.selected = position
            self.queued_actions.pop(
                self.current_player,
                None,
            )

            self.message = (
                f"Selected {position} with "
                f"{tile.army} units."
            )

            return

        if self.selected is None:
            self.message = (
                "Select one of the current player's "
                "owned tiles first."
            )
            return

        source_x, source_y = self.selected
        target_x, target_y = position

        distance = (
            abs(source_x - target_x)
            + abs(source_y - target_y)
        )

        if distance != 1:
            self.message = (
                "The destination must be orthogonally "
                "adjacent."
            )
            return

        action = Action(
            source=self.selected,
            destination=position,
            half=self.use_half,
        )

        self.queued_actions[
            self.current_player
        ] = action

        self.message = (
            f"{self.PLAYER_NAMES[self.current_player]} "
            "move queued. Press Enter to submit."
        )

    def submit_current_action(
        self,
        action: Optional[Action] = None,
    ):
        if self.environment.game_over:
            return

        if action is None:
            action = self.queued_actions.get(
                self.current_player,
                Action.pass_action(),
            )

        submitting_player = self.current_player

        result = self.environment.submit_action(
            submitting_player,
            action,
        )

        if not result["accepted"]:
            self.message = (
                f"Submission rejected: "
                f"{result.get('reason', 'unknown')}."
            )
            return

        self.selected = None
        self.queued_actions.pop(
            submitting_player,
            None,
        )

        if result["turn_resolved"]:
            # All engine input fields have now been cleared.
            self.queued_actions.clear()

            if self.environment.game_over:
                if self.environment.draw:
                    self.message = (
                        "The match ended in a draw."
                    )
                else:
                    self.message = (
                        f"{self.PLAYER_NAMES[self.environment.winner]} "
                        "won the game."
                    )
            else:
                self.message = (
                    f"Turn {self.environment.turn} resolved. "
                    "All input fields were reset."
                )

            self.current_player = self._first_pending_player()

        else:
            waiting_for = result["waiting_for"]

            self.message = (
                "Action accepted. Waiting for players: "
                + ", ".join(
                    str(player_id + 1)
                    for player_id in waiting_for
                )
            )

            self.current_player = self._first_pending_player()

    def _first_pending_player(self) -> int:
        pending = self.environment.get_pending_players()

        if pending:
            return pending[0]

        active = self.environment.get_active_players()

        if active:
            return active[0]

        return 0

    def draw_castle_icon(
        self,
        rectangle: pygame.Rect,
        color: tuple[int, int, int],
    ):
        padding = max(
            3,
            rectangle.width // 5,
        )

        castle_rectangle = pygame.Rect(
            rectangle.left + padding,
            rectangle.top + padding,
            rectangle.width - 2 * padding,
            rectangle.height - 2 * padding,
        )

        pygame.draw.rect(
            self.screen,
            color,
            castle_rectangle,
            2,
        )

        pygame.draw.line(
            self.screen,
            color,
            (
                castle_rectangle.left,
                castle_rectangle.top,
            ),
            (
                castle_rectangle.left,
                castle_rectangle.top - 3,
            ),
            2,
        )

        pygame.draw.line(
            self.screen,
            color,
            (
                castle_rectangle.right,
                castle_rectangle.top,
            ),
            (
                castle_rectangle.right,
                castle_rectangle.top - 3,
            ),
            2,
        )

    def draw_tile(
        self,
        position: tuple[int, int],
        rectangle: pygame.Rect,
        visible: set[tuple[int, int]],
    ):
        tile = self.environment.tile(position)

        x, y = position

        known_terrain = (
            self.environment.known_terrain[
                self.current_player
            ][y][x]
        )

        hidden = (
            self.show_fog
            and not self.spectator_mode
            and position not in visible
        )

        if hidden:
            if known_terrain == UNKNOWN:
                color = self.UNKNOWN_COLOR

            elif known_terrain == int(
                Terrain.MOUNTAIN
            ):
                color = self.MOUNTAIN_COLOR

            else:
                color = (38, 43, 51)

            pygame.draw.rect(
                self.screen,
                color,
                rectangle,
            )

            if known_terrain == int(Terrain.CASTLE):
                self.draw_castle_icon(
                    rectangle,
                    self.MUTED_TEXT_COLOR,
                )

            pygame.draw.rect(
                self.screen,
                self.GRID_COLOR,
                rectangle,
                1,
            )

            return

        base_color = self.PLAIN_COLOR

        if tile.terrain == Terrain.MOUNTAIN:
            base_color = self.MOUNTAIN_COLOR

        elif tile.terrain == Terrain.CASTLE:
            base_color = self.CASTLE_COLOR

        pygame.draw.rect(
            self.screen,
            base_color,
            rectangle,
        )

        if tile.owner in PLAYER_IDS:
            overlay = pygame.Surface(
                rectangle.size,
                pygame.SRCALPHA,
            )

            overlay.fill(
                (
                    *self.PLAYER_COLORS[tile.owner],
                    120,
                )
            )

            self.screen.blit(
                overlay,
                rectangle.topleft,
            )

        if tile.terrain == Terrain.MOUNTAIN:
            pygame.draw.polygon(
                self.screen,
                (105, 111, 120),
                [
                    (
                        rectangle.left + 3,
                        rectangle.bottom - 4,
                    ),
                    (
                        rectangle.centerx,
                        rectangle.top + 4,
                    ),
                    (
                        rectangle.right - 3,
                        rectangle.bottom - 4,
                    ),
                ],
            )

        elif tile.terrain == Terrain.CASTLE:
            self.draw_castle_icon(
                rectangle,
                (45, 38, 25),
            )

        elif tile.terrain == Terrain.GENERAL:
            radius = max(
                4,
                rectangle.width // 5,
            )

            pygame.draw.circle(
                self.screen,
                (255, 234, 125),
                rectangle.center,
                radius,
            )

            pygame.draw.circle(
                self.screen,
                (45, 38, 25),
                rectangle.center,
                radius,
                2,
            )

        if (
            tile.terrain != Terrain.MOUNTAIN
            and (
                tile.army > 0
                or tile.owner != NEUTRAL
            )
        ):
            army_text = self.small_font.render(
                str(tile.army),
                True,
                self.TEXT_COLOR,
            )

            self.screen.blit(
                army_text,
                army_text.get_rect(
                    center=rectangle.center
                ),
            )

        pygame.draw.rect(
            self.screen,
            self.GRID_COLOR,
            rectangle,
            1,
        )

    def draw_sidebar(
        self,
        sidebar_x: int,
    ):
        y = 18

        title = self.large_font.render(
            "GENERALS ARENA",
            True,
            self.TEXT_COLOR,
        )

        self.screen.blit(
            title,
            (sidebar_x, y),
        )

        y += 44

        environment_lines = [
            f"Turn: {self.environment.turn} / "
            f"{self.environment.max_turns}",
            (
                f"Map: {self.environment.width} x "
                f"{self.environment.height}"
            ),
            f"Seed: {self.environment.seed}",
            (
                "Move: HALF"
                if self.use_half
                else "Move: ALL BUT ONE"
            ),
            (
                "View: SPECTATOR"
                if self.spectator_mode
                else (
                    f"View: PLAYER "
                    f"{self.current_player + 1}"
                )
            ),
        ]

        for line in environment_lines:
            text = self.small_font.render(
                line,
                True,
                self.TEXT_COLOR,
            )

            self.screen.blit(
                text,
                (sidebar_x, y),
            )

            y += 21

        y += 10

        for player_id in PLAYER_IDS:
            territory, army, castles = (
                self.environment.get_stats(player_id)
            )

            submitted = (
                player_id
                in self.environment.pending_actions
            )

            eliminated = (
                player_id
                in self.environment.eliminated_players
            )

            if eliminated:
                status = "ELIMINATED"
            elif submitted:
                status = "SUBMITTED"
            else:
                status = "WAITING"

            name_text = self.font.render(
                (
                    f"{self.PLAYER_NAMES[player_id]} "
                    f"[{status}]"
                ),
                True,
                self.PLAYER_COLORS[player_id],
            )

            self.screen.blit(
                name_text,
                (sidebar_x, y),
            )

            y += 23

            stats_text = self.small_font.render(
                (
                    f"Tiles {territory}  "
                    f"Army {army}  "
                    f"Castles {castles}"
                ),
                True,
                self.TEXT_COLOR,
            )

            self.screen.blit(
                stats_text,
                (sidebar_x, y),
            )

            y += 27

        y += 8

        controls = [
            "CONTROLS",
            "1 / 2 / 3: select player",
            "Click: select / target",
            "H: half / full move",
            "Enter: submit queued action",
            "Space: submit pass",
            "Backspace: withdraw action",
            "F: toggle fog",
            "V: spectator view",
            "R: restart match",
            "Esc: quit",
        ]

        for index, line in enumerate(controls):
            color = (
                self.TEXT_COLOR
                if index == 0
                else self.MUTED_TEXT_COLOR
            )

            text = self.small_font.render(
                line,
                True,
                color,
            )

            self.screen.blit(
                text,
                (sidebar_x, y),
            )

            y += 20

    def draw(self):
        self.screen.fill(self.BACKGROUND)

        (
            offset_x,
            offset_y,
            cell_size,
            panel_width,
        ) = self.calculate_layout()

        visible = self.environment.get_visible_cells(
            self.current_player
        )

        for y in range(self.environment.height):
            for x in range(
                self.environment.width
            ):
                rectangle = pygame.Rect(
                    offset_x + x * cell_size,
                    offset_y + y * cell_size,
                    cell_size,
                    cell_size,
                )

                self.draw_tile(
                    (x, y),
                    rectangle,
                    visible,
                )

        if self.selected is not None:
            selected_x, selected_y = self.selected

            pygame.draw.rect(
                self.screen,
                self.SELECTION_COLOR,
                pygame.Rect(
                    offset_x
                    + selected_x * cell_size,
                    offset_y
                    + selected_y * cell_size,
                    cell_size,
                    cell_size,
                ),
                3,
            )

        queued_action = self.queued_actions.get(
            self.current_player
        )

        if (
            queued_action is not None
            and queued_action.destination is not None
        ):
            destination_x, destination_y = (
                queued_action.destination
            )

            pygame.draw.rect(
                self.screen,
                self.QUEUED_COLOR,
                pygame.Rect(
                    offset_x
                    + destination_x * cell_size,
                    offset_y
                    + destination_y * cell_size,
                    cell_size,
                    cell_size,
                ),
                3,
            )

        screen_width, screen_height = (
            self.screen.get_size()
        )

        sidebar_x = max(
            offset_x
            + self.environment.width
            * cell_size
            + 20,
            screen_width - panel_width + 10,
        )

        self.draw_sidebar(sidebar_x)

        message_bar = pygame.Rect(
            0,
            screen_height - 42,
            screen_width,
            42,
        )

        pygame.draw.rect(
            self.screen,
            (12, 15, 20),
            message_bar,
        )

        message_text = self.small_font.render(
            self.message[:150],
            True,
            self.TEXT_COLOR,
        )

        self.screen.blit(
            message_text,
            (
                16,
                message_bar.top + 13,
            ),
        )

        pygame.display.flip()

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_1:
                        self.switch_player(0)

                    elif event.key == pygame.K_2:
                        self.switch_player(1)

                    elif event.key == pygame.K_3:
                        self.switch_player(2)

                    elif event.key == pygame.K_h:
                        self.use_half = not self.use_half

                        queued = self.queued_actions.get(
                            self.current_player
                        )

                        if queued is not None:
                            self.queued_actions[
                                self.current_player
                            ] = Action(
                                source=queued.source,
                                destination=(
                                    queued.destination
                                ),
                                half=self.use_half,
                            )

                    elif event.key == pygame.K_f:
                        self.show_fog = (
                            not self.show_fog
                        )

                    elif event.key == pygame.K_v:
                        self.spectator_mode = (
                            not self.spectator_mode
                        )

                    elif event.key == pygame.K_SPACE:
                        self.submit_current_action(
                            Action.pass_action()
                        )

                    elif event.key == pygame.K_RETURN:
                        self.submit_current_action()

                    elif (
                        event.key
                        == pygame.K_BACKSPACE
                    ):
                        withdrawn = (
                            self.environment.withdraw_action(
                                self.current_player
                            )
                        )

                        if withdrawn:
                            self.message = (
                                "Submitted action withdrawn."
                            )
                        else:
                            self.message = (
                                "No submitted action to withdraw."
                            )

                    elif event.key == pygame.K_r:
                        self.restart()

                elif (
                    event.type
                    == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                ):
                    position = self.point_to_cell(
                        event.pos
                    )

                    if position is not None:
                        self.handle_click(position)

            self.draw()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# EXTERNAL AI EXAMPLE
# ============================================================

def example_external_ai_interaction():
    """
    This function is not run automatically.

    It demonstrates how an external controller can communicate with
    the environment without using Pygame.
    """

    environment = GeneralsEnvironment(seed=12345)

    while not environment.game_over:
        for player_id in environment.get_pending_players():
            observation = environment.get_observation(
                player_id
            )

            legal_actions = environment.get_legal_actions(
                player_id
            )

            # Replace this with an AI model decision.
            selected_action = legal_actions[0]

            result = environment.submit_action(
                player_id,
                selected_action,
            )

            if result["turn_resolved"]:
                print(
                    "Resolved turn:",
                    environment.turn,
                )