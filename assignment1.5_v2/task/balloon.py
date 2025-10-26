import argparse
import importlib.util
import os
import random
import sys
from math import cos, pi, sin, sqrt
from pathlib import Path
from random import randrange
from typing import Optional, Union

import numpy as np
import pygame
from pygame.locals import *

from task.human_player import HumanPlayer


def correct_path(current_path):
    """
    This function is used to get the correct path to the assets folder
    """
    return os.path.join(os.path.dirname(__file__), current_path)


GRAVITY = 0.09 * 3600
L = 25


def balloon(fps, players, mass=1.0):
    """
    Runs the balloon game.
    """
    # Game constants
    FPS = fps
    ALGO_FPS = 60 / 20
    WIDTH = 800
    HEIGHT = 800
    DT = 1 / 60
    gravity = GRAVITY
    arm = L

    # Initialize Pygame, load sprites
    FramePerSec = pygame.time.Clock()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Loading player and target sprites
    player_width = 80
    player_animation_speed = 0.3
    player_animation = []
    player_colors = ["red", "yellow", "blue", "green"]
    for anim_id in range(4):
        anim = []
        color = player_colors[anim_id]
        for i in range(1, 5):
            image = pygame.image.load(
                correct_path(
                    f"assets/images/objects/drone-sprites/{color}/drone-{i}.png"
                )
            )
            image.convert()
            anim.append(
                pygame.transform.scale(image, (player_width, int(player_width * 0.30)))
            )
        player_animation.append(anim)

    target_width = 30
    target_animation_speed = 0.1
    target_animation = []
    target_colors = ["red-plain", "yellow-plain", "blue-plain", "green-plain"]
    for anim_id in range(4):
        anim = []
        color = target_colors[anim_id]
        for i in range(1, 8):
            image = pygame.image.load(
                correct_path(
                    f"assets/images/balloon-sprites/{color}/{color}-{i}.png"
                )
            )
            image.convert()
            anim.append(
                pygame.transform.scale(image, (target_width, int(target_width * 1.73)))
            )
        target_animation.append(anim)

    # Loading background sprites
    cloud1 = pygame.image.load(
        correct_path(
            "assets/images/background-elements/cloud-1.png"
        )
    )
    cloud2 = pygame.image.load(
        correct_path(
            "assets/images/background-elements/cloud-2.png"
        )
    )
    sun = pygame.image.load(
        correct_path("assets/images/background-elements/sun.png")
    )
    cloud1.set_alpha(124)
    x_cloud1, y_cloud1, speed_cloud1 = (150, 200, 0.3)
    cloud2.set_alpha(124)
    x_cloud2, y_cloud2, speed_cloud2 = (400, 500, -0.2)
    sun.set_alpha(124)

    # Loading fonts
    pygame.font.init()
    name_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 20)
    name_hud_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 15)
    time_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 30)
    score_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Regular.ttf"), 20)
    respawn_timer_font = pygame.font.Font(
        correct_path("assets/fonts/Roboto-Bold.ttf"), 90
    )
    respawning_font = pygame.font.Font(
        correct_path("assets/fonts/Roboto-Regular.ttf"), 15
    )

    # Function to display info about a player

    def display_info(position):
        name_text = name_font.render(player.name, True, (255, 255, 255))
        screen.blit(name_text, (position, 20))
        target_text = score_font.render(
            "Score : " + str(player.target_counter), True, (255, 255, 255)
        )
        screen.blit(target_text, (position, 45))
        if player.dead == True:
            respawning_text = respawning_font.render(
                "Respawning...", True, (255, 255, 255)
            )
            screen.blit(respawning_text, (position, 70))

    # Initialize game variables
    time = 0
    step = 0
    time_limit = 100
    respawn_timer_max = 3

    # Generate 100 targets
    targets = []
    for i in range(100):
        targets.append((randrange(200, 600), randrange(200, 600)))

    # Game loop
    while True:
        pygame.event.get()

        # Display background
        screen.fill((131, 176, 181))

        x_cloud1 += speed_cloud1
        if x_cloud1 > WIDTH:
            x_cloud1 = -cloud1.get_width()
        screen.blit(cloud1, (x_cloud1, y_cloud1))

        x_cloud2 += speed_cloud2
        if x_cloud2 < -cloud2.get_width():
            x_cloud2 = WIDTH
        screen.blit(cloud2, (x_cloud2, y_cloud2))

        screen.blit(sun, (630, -100))

        time += DT
        step += 1

        # For each player
        for player_index, player in enumerate(players):
            if player.dead:
                player.prev_target_counter = -1
                if not hasattr(player, "thruster_left"):
                    player.thruster_left = 0
                    player.thruster_right = 0

            if not player.dead:
                # Calculate propeller force in function of input
                if player.name == "PID":
                    thruster_left, thruster_right = player.act(
                        [
                            targets[player.target_counter][0],
                            targets[player.target_counter][1],
                            player.x_position,
                            player.y_position,
                            player.x_speed,
                            player.y_speed,
                            player.angle,
                            player.angular_speed,
                        ]
                    )
                elif "iLQR" in player.name:
                    if ALGO_FPS == 1 or step % ALGO_FPS == 1:
                        player.thruster_left, player.thruster_right = player.act(
                            target_pos=np.array(
                                [
                                    targets[player.target_counter][0],
                                    targets[player.target_counter][1],
                                ]
                            ),
                            obs=np.array(
                                [
                                    player.x_position,
                                    player.y_position,
                                    player.angle,
                                    player.x_speed,
                                    player.y_speed,
                                    player.angular_speed,
                                ]
                            ),
                        )
                    elif not hasattr(player, "thruster_left"):
                        player.thruster_left = 0
                        player.thruster_right = 0
                    thruster_left = player.thruster_left
                    thruster_right = player.thruster_right
                else:
                    thruster_left, thruster_right = player.act([])

                # Calculate position
                player.x_position += player.x_speed * DT
                player.y_position += player.y_speed * DT
                player.angle += player.angular_speed * DT

                # Calculate accelerations according to Newton's laws of motion
                player.x_acceleration = (
                    -(thruster_left + thruster_right)
                    * sin(player.angle * pi / 180)
                    / mass
                )
                player.y_acceleration = (
                    -(thruster_left + thruster_right)
                    * cos(player.angle * pi / 180)
                    / mass
                    + gravity
                )
                player.angular_acceleration = (
                    arm * (thruster_right - thruster_left) / mass
                )

                # Calculate speed
                player.x_speed += player.x_acceleration * DT
                player.y_speed += player.y_acceleration * DT
                player.angular_speed += player.angular_acceleration * DT

                # Calculate distance to target
                dist = sqrt(
                    (player.x_position - targets[player.target_counter][0]) ** 2
                    + (player.y_position - targets[player.target_counter][1]) ** 2
                )

                # If target reached, respawn target
                if dist < 50:
                    player.target_counter += 1

                # If to far, die and respawn after timer
                elif dist > 1000:
                    player.dead = True
                    player.respawn_timer = respawn_timer_max
            else:
                # Display respawn timer
                if player.name == "Human":
                    respawn_text = respawn_timer_font.render(
                        str(int(player.respawn_timer) + 1), True, (255, 255, 255)
                    )
                    respawn_text.set_alpha(124)
                    screen.blit(
                        respawn_text,
                        (
                            WIDTH / 2 - respawn_text.get_width() / 2,
                            HEIGHT / 2 - respawn_text.get_height() / 2,
                        ),
                    )

                player.respawn_timer -= DT
                # Respawn
                if player.respawn_timer < 0:
                    player.dead = False
                    (
                        player.angle,
                        player.angular_speed,
                        player.angular_acceleration,
                    ) = (
                        0,
                        0,
                        0,
                    )
                    (player.x_position, player.x_speed, player.x_acceleration) = (
                        400,
                        0,
                        0,
                    )
                    (player.y_position, player.y_speed, player.y_acceleration) = (
                        400,
                        0,
                        0,
                    )

            # Display target and player
            target_sprite = target_animation[player.anim_id][
                int(step * target_animation_speed) % len(target_animation)
            ]
            target_sprite.set_alpha(player.alpha)
            screen.blit(
                target_sprite,
                (
                    targets[player.target_counter][0]
                    - int(target_sprite.get_width() / 2),
                    targets[player.target_counter][1]
                    - int(target_sprite.get_height() / 2),
                ),
            )

            player_sprite = player_animation[player.anim_id][
                int(step * player_animation_speed) % len(player_animation)
            ]
            player_copy = pygame.transform.rotate(player_sprite, player.angle)
            player_copy.set_alpha(player.alpha)
            screen.blit(
                player_copy,
                (
                    player.x_position - int(player_copy.get_width() / 2),
                    player.y_position - int(player_copy.get_height() / 2),
                ),
            )

            # Display player name
            name_hud_text = name_hud_font.render(player.name, True, (255, 255, 255))
            screen.blit(
                name_hud_text,
                (
                    player.x_position - int(name_hud_text.get_width() / 2),
                    player.y_position - 30 - int(name_hud_text.get_height() / 2),
                ),
            )

            # Display player info
            if player_index == 0:
                display_info(20)
            elif player_index == 1:
                display_info(130)
            elif player_index == 2:
                display_info(240)
            elif player_index == 3:
                display_info(350)

            time_text = time_font.render(
                "Time : " + str(int(time_limit - time)), True, (255, 255, 255)
            )
            screen.blit(time_text, (670, 30))

        # Ending conditions
        if time > time_limit:
            break

        pygame.display.update()
        FramePerSec.tick(FPS)

    # Print scores and who won
    print("")
    scores = []
    for player in players:
        print(player.name + " collected : " + str(player.target_counter))
        scores.append(player.target_counter)
    winner = players[np.argmax(scores)].name

    print("")
    print("Winner is : " + winner + " !")


def import_module_from_path(path: Union[str, Path], module_name: Optional[str] = None):
    path = Path(path)
    if module_name is None:
        module_name = path.stem  # default: filename without extension
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fast", action="store_true", help="Run a fast experiment.")
    parser.add_argument(
        "--mass", type=float, default=1.0, help="Mass of the quad. Default: 1"
    )
    parser.add_argument("--human", action="store_true", help="Add Human player.")
    parser.add_argument("--pid", action="store_true", help="Add PID player.")
    parser.add_argument("--ilqr", action="store_true", help="Add iLQR player.")
    parser.add_argument("--mpc", action="store_true", help="Use iLQR+MPC.")
    parser.add_argument(
        "--seed", type=int, help="Random seed. By default 0x42", default=0x42
    )
    parser.add_argument(
        "--player_path", type=str, help="Path to root with player files."
    )
    args = parser.parse_args()

    m = args.mass
    Ixx = m
    dt = 1 / 20

    players = []

    if args.player_path:
        ilqr_player = import_module_from_path(Path(args.player_path) / "ilqr_player.py")
        pid_player = import_module_from_path(Path(args.player_path) / "pid_player.py")
        iLQRPlayer = ilqr_player.iLQRPlayer
        PIDPlayer = pid_player.PIDPlayer
    else:
        from task.ilqr_player import iLQRPlayer
        from task.pid_player import PIDPlayer

    if args.pid:
        players += [PIDPlayer()]
    if args.ilqr:
        players += [iLQRPlayer(dt, m, GRAVITY, Ixx, L, use_mpc=False)]
    if args.mpc:
        players += [iLQRPlayer(dt, m, GRAVITY, Ixx, L, use_mpc=True)]
    if args.human or len(players) == 0:
        players += [HumanPlayer()]

    if args.fast:
        fps = 600
    else:
        fps = 60

    random.seed(args.seed)
    balloon(fps, players, args.mass)
