import gym
import numpy as np
import os
import gc
import pandas as pd
import random
import time
import carla
import psutil, subprocess, signal
import json
import math

from scenic.simulators.carla.simulator import CarlaSimulator
from scenic.core.simulators import SimulationCreationError
from scenic.core.scenarios import Scenario
from scenic.syntax.translator import scenarioFromFile
from collections import OrderedDict
from scenic.syntax.translator import InvalidScenarioError
from scenic.core.simulators import EndSimulationAction
import scenic.syntax.veneer as veneer
import scenic.syntax.translator as translator
from scenic.syntax.veneer import currentSimulation, currentScenario, runningScenarios
from scenic.syntax.veneer import currentBehavior, _globalParameters

from tools.logs import Logger
from distance_calculator import VehicleDistanceCalculator
import gym.utils.seeding


class ScenicCarlaEnv(gym.Env):
    def __init__(self, scenefile, ellipse_json_path, attack_type=None, max_steps=500, tm_port=None, replay_dir=None,
                 use_ellipse=True, reward_terms=["re1", "re2", "re3"]):
        super().__init__()
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        Logger.info(f"tm_port: {tm_port}")

        self.sensors = []
        self.max_steps = max_steps
        self.tm_port = tm_port
        self.replay_dir = replay_dir
        self.episode_counter = 0
        self.current_replay = []
        self.episode_obs = []
        self.total_cost = 0.0
        self.max_ellipse_norm = 6000
        self.max_cost = 3

        self.use_ellipse = use_ellipse
        self.reward_terms = reward_terms
        Logger.info(f"Ablation Config: use_ellipse={use_ellipse}, reward_terms={reward_terms}")

        self.ellipse_params = None
        if self.use_ellipse:
            with open(ellipse_json_path, 'r', encoding='utf-8') as f:
                self.ellipse_params = json.load(f)
            Logger.info(f"Loaded ellipse parameters: {self.ellipse_params}")

        self.attack_strength = {
            "Attack_carla10_Stealth_Fade": 0.5,
            "Attack_carla10_Stealth_Pulse": 0.7,
            "Attack_carla10_Stealth_Jitter": 1.0,
            "Attack_carla10_TimingDelay": 1.2,
            "Attack_carla10_Spoofing": 1.5,
            "None": 0.3
        }
        self.current_attack_type = attack_type if attack_type else "None"
        self.current_attack_strength = self.attack_strength.get(self.current_attack_type, 1.0)
        self.c_max = 2.5
        self.w = 1.0

        self.carla_ego=None
        self.adversary=None

        self.params = {
            "trafficManagerPort": tm_port
        }
        self.scenario_path = scenefile

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.simulator = CarlaSimulator(
            carla_map = "Town05",
            map_path = "/home/luliu/Project/Verification/VeriFal/Scenic_extension/tests/formats/opendrive/maps/CARLA/Town05.xodr",
            render=False,
            traffic_manager_port = self.tm_port
        )
        self.sim = None
        self.current_step_data = None

        self.sensor_reference_pool = []
        self.sensor_recycle_delay = 3

    def _start_new_simulation(self):
        try:

            try:
                if veneer.simulationInProgress():
                    veneer.endSimulation(currentSimulation)
                veneer.currentSimulation = None
                veneer.currentScenario = None
                veneer.runningScenarios = set()
                veneer.currentBehavior = None
                veneer._globalParameters = {}
            except Exception as e:
                Logger.warning(f"[Pre-scenario] Scenic cleanup failed: {e}")

            self.scenario = translator.scenarioFromFile(self.scenario_path, self.params)
            self.scene, _ = self.scenario.generate()

            Logger.info("createSimulation()")
            self.sim=self.simulator.createSimulation(self.scene,verbosity=3)
            Logger.info("tick()")
            self.simulator.world.tick()
            time.sleep(0.5)

            self.dynamic_scenario = self.sim.scene.dynamicScenario
            self.dynamic_scenario._start()
            for agent in self.sim.agents:
                if not agent.behavior._runningIterator:
                    agent.behavior._start(agent)

            Logger.info("_update_actors_from_scene()")
            self._update_actors_from_scene()
            self.sim.updateObjects()
            self.step_index = 0
            self.current_replay = []
            self.episode_obs = []
            self.episode_step_data = []

        except SimulationCreationError as e:
            Logger.error(f"[Simulation Init Error] {e}")
            raise e
        except Exception as e:
            Logger.error(f"[Unexpected Simulation Error] {e}")
            raise e

    def _cleanup_actors(self):
        if self.simulator and self.simulator.world:
            for actor in self.simulator.world.get_actors():
                if actor.type_id.startswith(("vehicle.", "sensor.")):
                    try:
                        actor.destroy()
                    except:
                        pass
        self.sensors.clear()
        self.carla_ego = None
        self.adversary = None
        gc.collect()

    def reset(self):
        Logger.info(f"[into reset...]")
        self.episode_counter += 1

        if hasattr(self, 'simulator') and self.simulator:
            self._cleanup_actors()
        else:
            Logger.info(f"[Reset] skipping...")

        self.step_index = 0
        self.total_cost = 0.0
        self.current_step_data = None
        self.episode_step_data = []
        self.episode_obs = []
        self.current_replay = []

        for attempt in range(3):
            try:
                self._start_new_simulation()
                break
            except RuntimeError as e:
                time.sleep(3)
        else:
            raise RuntimeError("fail...")

        try:
            self.sim.updateObjects()

            self.current_step_data = self.sim.step_data[0] if self.sim.step_data else None
            if not self.current_step_data:
                raise ValueError("step_data is none")

            self.episode_step_data.append(self.current_step_data)

            obs = self._get_observation_from_step_data()
            self.episode_obs.append(obs)
            return obs
        except Exception as e:
            return np.zeros((7,), dtype=np.float32)

    def step(self, action):
        veneer.beginSimulation(self.sim)
        Logger.info(f"[Step] Episode: {self.episode_counter}, Step: {self.step_index}, Action: {action}")
        self.step_index += 1
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        control = carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake
        )

        step_cost = (abs(steer) + brake) * 0.1
        step_cost = step_cost / 2.0
        self.total_cost += step_cost

        try:
            if self.carla_ego and self.carla_ego.is_alive:
                self.carla_ego.apply_control(control)
                carla_control = self.carla_ego.get_control()

            termination_reason = self.dynamic_scenario._step()
            if termination_reason:
                return self.reset(), -100, True, {"reason": termination_reason}

            all_actions = OrderedDict()
            schedule = self.sim.scheduleForAgents()
            for agent in schedule:
                actions = agent.behavior._step()

                if isinstance(actions, EndSimulationAction):
                    termination_reason = str(actions)
                    break
                if not self.sim.actionsAreCompatible(agent, actions):
                    raise InvalidScenarioError(f"Agent {agent} is no suitable: {actions}")
                all_actions[agent] = actions
            if termination_reason:
                return self.reset(), -100, True, {"reason": termination_reason}

            filtered_actions = OrderedDict()
            for agent, actions in all_actions.items():
                if agent.role == 'ego':
                    continue
                filtered_actions[agent] = actions

            self.sim.executeActions(filtered_actions)
            for agent, actions in filtered_actions.items():
                Logger.info(f"  {agent.role} : {[a.__class__.__name__ for a in actions]}")

            self.sim.step()

            self.sim.updateObjects()
            self.current_step_data = self.sim.step_data[-1] if self.sim.step_data else None
            if not self.current_step_data:
                raise ValueError("Scenic do not generate step_data")
            self.episode_step_data.append(self.current_step_data)

            obs = self._get_observation_from_step_data()
            self.episode_obs.append(obs)

            if obs is None or np.isnan(obs).any() or np.isinf(obs).any():
                Logger.error(f"[Step] Invalid observation detected: {obs}")
                return np.zeros_like(obs), -100, True, {"is_unsafe": True, "cost": 1e6}

            reward, done, info = self._compute_reward_done_info_from_step_data(obs, brake, steer, throttle)

            self._collect_trajectory_data(obs, action, control, reward, done, info)

            if done and self.replay_dir:
                self._save_episode_replay()
                self.current_replay = []

            return obs, reward, done, info
        except Exception as e:
            Logger.error(f"[Step Error] {e}")
            return self.reset(), -100, True, {"is_unsafe": True, "cost": 1e6}
        finally:
            veneer.endSimulation(self.sim)


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _update_actors_from_scene(self):
        self.carla_ego = None
        self.adversary = None
        self.sensors = []

        for obj in self.scene.objects:
            if not hasattr(obj, 'carlaActor') or obj.carlaActor is None:
                continue

            actor = obj.carlaActor
            actor_type = actor.type_id

            if actor_type.startswith('vehicle.'):
                role = getattr(obj, 'role', None)
                if role == "ego":
                    self.carla_ego = actor
                elif role == "adversary":
                    self.adversary = actor
            elif actor_type.startswith('sensor.') and actor.is_alive:
                self.sensors.append(actor)

        Logger.info(f"[Actor Binding] Ego: {self.carla_ego}, Adversary: {self.adversary}, Sensors: {len(self.sensors)}")

    def _get_observation_from_step_data(self):
        try:
            step_data = self.current_step_data
            if not step_data:
                step_data = self.sim.step_data[-1] if self.sim.step_data else None
            if not step_data:
                step_data = self._generate_initial_step_data()
                self.current_step_data = step_data

            ego_data = None
            adv_data = None
            for obj in step_data['objects']:
                if obj['role'] == 'ego':
                    ego_data = obj
                elif obj['role'] == 'adversary':
                    adv_data = obj
            if not ego_data:
                raise ValueError("step_data could not find ego")

            ego_props = ego_data['properties']
            ego_pos = ego_props['position']
            ego_vel = ego_props['velocity']
            ego_heading = ego_props['heading']

            adv_pos = (0.0, 0.0)
            adv_heading = 0.0
            if adv_data:
                adv_props = adv_data['properties']
                adv_pos = adv_props['position']
                adv_heading = adv_props['heading']

            rel_x = ego_pos[0] - adv_pos[0]
            rel_y = ego_pos[1] - adv_pos[1]

            min_distance = VehicleDistanceCalculator.calculate_min_distance(
                ego_pos=ego_pos,
                ego_heading=ego_heading,
                adv_pos=adv_pos,
                adv_heading=adv_heading
            )

            norm_in_ellipse = 0.0
            if self.use_ellipse and self.ellipse_params:
                ellipse_center = np.array(self.ellipse_params["center"])
                major_axis = self.ellipse_params["major_axis"]
                minor_axis = self.ellipse_params["minor_axis"]
                ellipse_angle_deg = self.ellipse_params["angle"]

                corrected_ellipse_angle_deg = ellipse_angle_deg - 90.0
                corrected_ellipse_angle_deg %= 360.0
                corrected_ellipse_angle_rad = math.radians(corrected_ellipse_angle_deg)

                ego_rel_center = np.array([
                    ego_pos[0] - ellipse_center[0],
                    ego_pos[1] - ellipse_center[1]
                ])

                rotation_matrix = np.array([
                    [math.cos(corrected_ellipse_angle_rad), math.sin(corrected_ellipse_angle_rad)],
                    [-math.sin(corrected_ellipse_angle_rad), math.cos(corrected_ellipse_angle_rad)]
                ])
                ego_rotated = rotation_matrix @ ego_rel_center

                norm_in_ellipse = (ego_rotated[0] / major_axis) ** 2 + (ego_rotated[1] / minor_axis) ** 2

            rel_angle = ego_heading - adv_heading
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

            obs = np.array([
                rel_x, rel_y,
                ego_vel[0], ego_vel[1],
                norm_in_ellipse,
                min_distance,
                rel_angle
            ], dtype=np.float32)

        except Exception as e:
            raise RuntimeError(f"{e}")

    def _compute_reward_done_info_from_step_data(self, obs, brake, steer, throttle):
        rel_x, rel_y, vx, vy, norm_in_ellipse, min_distance, rel_angle = obs
        ego_speed = np.linalg.norm([vx, vy])
        total_reward = 0.0
        Re1 = Re2 = Re3 = 0.0
        ellipse_cost = 0.0
        control_cost = 0.0

        if "re1" in self.reward_terms and self.use_ellipse:
            d = min(norm_in_ellipse / self.max_ellipse_norm, 2.0)
            if d > 1:
                Re1 = 50.0 * np.exp(-1.5 * d) * 2
            else:
                Re1 = 20.0
            total_reward += Re1
        else:
            d = float("inf")

        if "re2" in self.reward_terms:
            dr = min_distance
            if dr > 1.5:
                Re2 = 10 * np.exp(-0.01 * (dr - 1.5))
            else:
                Re2 = 20.0
            total_reward += Re2

        if "re3" in self.reward_terms:
            if min_distance <= 3.0:
                Re3 = - self.current_attack_strength * self.total_cost
            else:
                Re3 = 0.0

        done = False
        termination_reason = "not terminated"

        if "re1" in self.reward_terms and self.use_ellipse and d <= 1.0:
            done = True
            termination_reason = "success"

        elif min_distance < 0.1:
            done = True
            termination_reason = "success"

        elif self.total_cost > self.max_cost:
            done = True
            termination_reason = "cost limit"

        elif self.step_index >= self.max_steps:
            done = True
            termination_reason = "timeout"

        info = {
            "is_unsafe": min_distance < 1.5,
            "cost": self.total_cost,
            "is_success": termination_reason == "success",
            "termination_reason": termination_reason,
            "Re1": Re1,
            "Re2": Re2,
            "Re3": Re3,
            "min_distance": min_distance,
            "norm_in_ellipse": norm_in_ellipse
        }

        return total_reward, done, info

    def _generate_initial_step_data(self):
        step_data = {
            'step': 0,
            'time': 0.0,
            'objects': []
        }
        for obj in self.sim.objects:
            properties = obj._dynamicProperties
            values = self.sim.getProperties(obj, properties)
            obj_data = {
                'id': id(obj),
                'role': getattr(obj, 'role', 'unknown'),
                'type': type(obj).__name__,
                'properties': values,
                'control': {
                    'throttle': 0.0,
                    'brake': 0.0,
                    'steer': 0.0,
                    'gear': 0,
                    'hand_brake': False,
                    'reverse': False
                }
            }
            if hasattr(obj, 'carlaActor') and hasattr(obj.carlaActor, 'get_control'):
                carla_control = obj.carlaActor.get_control()
                obj_data['control'] = {
                    'throttle': carla_control.throttle,
                    'brake': carla_control.brake,
                    'steer': carla_control.steer,
                    'gear': carla_control.gear,
                    'hand_brake': carla_control.hand_brake,
                    'reverse': carla_control.reverse
                }
            step_data['objects'].append(obj_data)
        return step_data

    def _collect_trajectory_data(self, obs, action, control, reward, done, info):
        if self.carla_ego and self.carla_ego.is_alive:
            objects_data = []
            for obj in self.current_step_data['objects']:
                pos_str = str(obj['properties']['position']).strip('()')
                pos_coords = [float(coord.strip()) for coord in pos_str.split('@')]
                position = pos_coords[:2]

                vel_str = str(obj['properties']['velocity']).strip('()')
                vel_coords = [float(coord.strip()) for coord in vel_str.split('@')]
                velocity = vel_coords[:2]

                obj_data = {
                    'role': obj['role'],
                    'type': obj['type'],
                    'position':  position,
                    'velocity': velocity,
                    'speed': obj['properties']['speed'],
                    'heading': obj['properties']['heading']
                }
                objects_data.append(obj_data)

            ego_data = next(obj for obj in objects_data if obj['role'] == 'ego')

            done_py = bool(done)

            info_py = {}
            for key, value in info.items():
                if isinstance(value, np.bool_):
                    info_py[key] = bool(value)
                elif isinstance(value, np.generic):
                    info_py[key] = value.item()
                else:
                    info_py[key] = value

            step_trajectory = {
                'step': self.step_index,
                'timestamp': time.time(),
                'action': action.tolist(),
                'control': {
                    'steer': control.steer,
                    'throttle': control.throttle,
                    'brake': control.brake
                },
                'ego': ego_data,
                'objects': objects_data,
                'observation': obs.tolist(),
                'reward': reward,
                'done': done_py,
                'info': info_py
            }

            self.current_replay.append(step_trajectory)

    def _save_episode_replay(self):
        if self.replay_dir and self.current_replay:
            try:
                episode_num = self.episode_counter

                replay_subdir = os.path.join(self.replay_dir, "replays")
                os.makedirs(replay_subdir, exist_ok=True)
                replay_path = os.path.join(replay_subdir, f"episode_{episode_num}.json")

                with open(replay_path, 'w') as f:
                    json.dump(self.current_replay, f, indent=2)

                Logger.info(f"Replay saved to: {replay_path}")
            except Exception as e:
                Logger.error(f"Failed to save replay: {e}")
