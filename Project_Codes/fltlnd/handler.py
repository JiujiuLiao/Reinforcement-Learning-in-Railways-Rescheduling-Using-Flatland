import os
from typing import Optional
from fltlnd.utils import TrainingMode
import json
import time
import random
from flatland.envs import malfunction_generators as mal_gen
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from fltlnd.deadlocks import DeadlocksDetector
import fltlnd.agent as agent_classes
import fltlnd.obs as obs_classes
import fltlnd.logger as logger_classes
import fltlnd.replay_buffer as memory_classes
import fltlnd.predict as predictor_classes
from flatland.envs.agent_utils import RailAgentStatus

from fltlnd.action_masking import ActionMasker, MinimalActionMasker
from fltlnd.staggered_departures import StaggeredDepartureController, get_spacing_for_phase

class ExcHandler:
    def __init__(self, params: dict, training_mode: TrainingMode, rendering: bool, checkpoint: Optional[str],
                 synclog: bool, verbose: bool):
        self._sys_params = params['sys']
        self._obs_params = params['obs']
        self._trn_params = params['trn']
        self._log_params = params['log']

        self._rendering = rendering
        self._training = training_mode is not TrainingMode.EVAL
        self._tuning = training_mode is TrainingMode.TUNING
        self._train_best = training_mode in [TrainingMode.BEST, TrainingMode.EVAL]
        self._save_checkpoints = training_mode is not TrainingMode.DEBUG
        self._verbose = verbose

        self._default_checkpoint = checkpoint

        self._obs_class = getattr(obs_classes, self._sys_params['obs_class'])
        self._agent_class = getattr(agent_classes, self._sys_params['agent_class'])
        self._logger_class = getattr(logger_classes, self._sys_params['log_class'])
        if self._sys_params['agent_class'] == "PPOAgent":
           self._memory_class = memory_classes.AgentEpisodeBuffer
        else:
           self._memory_class = getattr(memory_classes, self._sys_params['memory_class'])
        
        if self._sys_params['predictor_class'] != None:
            self._predictor_class = getattr(predictor_classes, self._sys_params['predictor_class'])
            self._predictor = self._predictor_class(self._obs_params)
        else:
            self._predictor = None

        self._obs_wrapper = self._obs_class(self._obs_params, self._predictor)
        self._env_handler = EnvHandler(self._sys_params['base_dir'] + "parameters/environments.json",
                                       self._obs_wrapper.builder,
                                       self._rendering)
        self._obs_wrapper.env = self._env_handler

        self._action_size = 5
        self._state_size = self._obs_wrapper.get_state_size() + self._obs_wrapper.n_global_features

        self._logger = self._logger_class(self._sys_params['base_dir'], self._log_params, self._tuning, synclog)

        # ============================================================
        # ACTION MASKING: DISABLED
        # 
        # We disable action masking to let the agent learn coordination
        # naturally through rewards/penalties. The MinimalActionMasker
        # was still causing issues.
        # ============================================================
        self._action_masker = MinimalActionMasker()  # Keep for reference but don't use
        self._use_action_masking = False  # DISABLED!
        
        # ============================================================
        # STAGGERED DEPARTURES - with LARGER spacing
        # ============================================================
        self._departure_controller = None
        self._current_step = 0
        self._use_staggered_departures = True

    def _get_action(self, obs, agent_handle):
        """
        Get action WITHOUT masking - let agent learn from consequences.
        """
        return self._policy.act(obs)

    def start(self, n_episodes):
        start_time = time.time()
        random.seed(self._sys_params['seed'])
        np.random.seed(self._sys_params['seed'])

        for run_id, params in enumerate(self._logger.get_run_params()):
            self._trn_params.update(params)
            self._policy = self._agent_class(self._state_size, self._action_size, self._trn_params,
                                             self._memory_class, self._training, self._train_best,
                                             self._sys_params['base_dir'],
                                             self._default_checkpoint)
            self._logger.run_start(self._trn_params, str(self._policy))
            self._env_handler.update(self._trn_params['env'], self._sys_params['seed'])

            self._max_steps = int(4 * 2 * (self._env_handler._params['x_dim'] + self._env_handler._params['y_dim'] + (
                    self._env_handler.get_num_agents() / self._env_handler._params['n_cities'])))

            # ============================================================
            # STAGGERED DEPARTURES: MUCH LARGER spacing for 5 agents
            # ============================================================
            n_agents = self._env_handler.get_num_agents()
            if self._use_staggered_departures and n_agents > 1:
                # INCREASED spacing: 40 steps between departures for 5 agents
                # This means agent 4 departs at step 160, giving LOTS of room
                if n_agents >= 5:
                    departure_spacing = 40  # Much larger!
                elif n_agents >= 3:
                    departure_spacing = 25
                else:
                    departure_spacing = 15
                    
                self._departure_controller = StaggeredDepartureController(
                    n_agents=n_agents,
                    departure_spacing=departure_spacing,
                    spacing_mode='fixed'
                )
                print(f"[StaggeredDepartures] ENABLED: spacing={departure_spacing} for {n_agents} agents")
                print(f"[StaggeredDepartures] Departure times: {self._departure_controller.departure_times}")
                print(f"[StaggeredDepartures] Last agent departs at step {(n_agents-1) * departure_spacing}")
            else:
                self._departure_controller = None
                if n_agents > 1:
                    print(f"[StaggeredDepartures] DISABLED for {n_agents} agents")
            # ============================================================

            eval_score = None
            for episode_idx in range(n_episodes):
                self._policy.episode_start()

                score = 0
                action_dict = dict()
                action_count = [0] * self._action_size
                agent_obs = [None] * self._env_handler.get_num_agents()
                agent_prev_obs = [None] * self._env_handler.get_num_agents()
                agent_prev_action = [2] * self._env_handler.get_num_agents()
                update_values = False

                obs, info = self._env_handler.reset()
                
                # Reset step counter and departure controller
                self._current_step = 0
                if self._departure_controller is not None:
                    self._departure_controller.reset()
                
                # Reset observation wrapper
                if hasattr(self._obs_wrapper, 'reset'):
                    self._obs_wrapper.reset()
                
                # Tracking variables
                deadlocked_agents = set()
                stuck_agents = set()
                agent_last_distance = {}
                agent_no_progress_counter = {}
                NO_PROGRESS_THRESHOLD = 50
                
                self._prev_distances = {}
                self._agent_deadlock_steps = {}
                self._agent_stationary_steps = {}
                self._agent_last_positions = {}
                self._agent_collision_count = {}
                
                # Track which agents have completed
                self._completed_agents = set()
                
                num_agents = self._env_handler.get_num_agents()

                for agent in self._env_handler.get_agents_handle():
                    if obs[agent]:
                        agent_obs[agent] = self._obs_wrapper.normalize(obs[agent], self._env_handler, agent)
                        agent_prev_obs[agent] = agent_obs[agent].copy()

                count_steps = 0
                for step in range(self._max_steps - 1):
                    count_steps += 1

                    act_time = time.time()
                    
                    # ============================================================
                    # Action selection WITHOUT masking
                    # ============================================================
                    for agent in self._env_handler.get_agents_handle():
                        agent_obj = self._env_handler.env.agents[agent]
                        
                        if info['action_required'][agent]:
                            update_values = True
                            # NO MASKING - let agent learn naturally
                            action = self._policy.act(agent_obs[agent])
                            action_count[action] += 1
                        else: 
                            update_values = False
                            if agent_obj.status == RailAgentStatus.ACTIVE:
                                action = 2  # FORWARD
                            else:
                                action = 0  # DO_NOTHING
                        
                        action_dict.update({agent: action})
                    
                    # ============================================================
                    # STAGGERED DEPARTURES: Apply departure mask
                    # This is the ONLY masking we do
                    # ============================================================
                    if self._departure_controller is not None:
                        agent_states = {
                            a: self._env_handler.env.agents[a].status.value 
                            for a in self._env_handler.get_agents_handle()
                        }
                        action_dict = self._departure_controller.mask_actions_for_departure(
                            action_dict,
                            self._current_step,
                            agent_states
                        )
                    self._current_step += 1
                    
                    act_time = time.time() - act_time

                    next_obs, all_rewards, done, info = self._env_handler.step(action_dict)

                    # ============================================================
                    # NEW REWARD SHAPING V2
                    # 
                    # Key principles:
                    # 1. STRONGLY reward FORWARD movement
                    # 2. PENALIZE STOP action (unless near completion)
                    # 3. Large completion bonus
                    # 4. Moderate deadlock penalty (not too harsh)
                    # ============================================================
                    train_time = time.time()
                    
                    for agent in self._env_handler.get_agents_handle():
                        shaped_reward = 0.0
                        agent_obj = self._env_handler.env.agents[agent]
                        action_taken = action_dict.get(agent, 0)
                        
                        # Skip non-active agents
                        if agent_obj.status not in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART]:
                            if done[agent] and agent_obj.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                                shaped_reward = 1.0
                                self._completed_agents.add(agent)
                            
                            if self._training and agent_prev_obs[agent] is not None:
                                self._policy.step(
                                    agent_prev_obs[agent],
                                    agent_prev_action[agent],
                                    shaped_reward,
                                    agent_obs[agent] if agent_obs[agent] is not None else agent_prev_obs[agent],
                                    done[agent],
                                    agent
                                )
                            score += shaped_reward
                            continue
                        
                        # --------------------------------------------------------
                        # REWARD 1: Progress toward target (INCREASED)
                        # --------------------------------------------------------
                        if agent_obj.position is not None and agent_obj.target is not None:
                            current_dist = abs(agent_obj.position[0] - agent_obj.target[0]) + \
                                          abs(agent_obj.position[1] - agent_obj.target[1])
                            
                            map_diagonal = self._env_handler.x_dim + self._env_handler.y_dim
                            prev_dist = self._prev_distances.get(agent, current_dist)
                            
                            dist_change = prev_dist - current_dist
                            
                            if dist_change > 0:
                                # INCREASED reward for getting closer
                                shaped_reward += 0.3 * (dist_change / map_diagonal)
                            elif dist_change < 0:
                                # Penalty for moving away
                                shaped_reward += 0.2 * (dist_change / map_diagonal)
                            
                            self._prev_distances[agent] = current_dist
                        
                        # --------------------------------------------------------
                        # REWARD 2: Action-based rewards
                        # Encourage FORWARD, discourage excessive STOP
                        # --------------------------------------------------------
                        if action_taken == 2:  # MOVE_FORWARD
                            shaped_reward += 0.02  # Small bonus for moving forward
                        elif action_taken == 4:  # STOP_MOVING
                            # Only penalize STOP if agent is not near target
                            if agent_obj.position is not None and agent_obj.target is not None:
                                dist_to_target = abs(agent_obj.position[0] - agent_obj.target[0]) + \
                                               abs(agent_obj.position[1] - agent_obj.target[1])
                                if dist_to_target > 3:  # Not near target
                                    shaped_reward -= 0.03  # Penalty for unnecessary stopping
                        
                        # --------------------------------------------------------
                        # REWARD 3: Stationary penalty (simplified)
                        # --------------------------------------------------------
                        current_pos = agent_obj.position
                        last_pos = self._agent_last_positions.get(agent)
                        
                        if current_pos is not None:
                            # Check if waiting for staggered departure (no penalty)
                            waiting_for_departure = False
                            if self._departure_controller is not None:
                                if agent_obj.status == RailAgentStatus.READY_TO_DEPART:
                                    if not self._departure_controller.can_depart(agent, self._current_step - 1):
                                        waiting_for_departure = True
                            
                            if last_pos == current_pos and not waiting_for_departure:
                                self._agent_stationary_steps[agent] = self._agent_stationary_steps.get(agent, 0) + 1
                                stationary_steps = self._agent_stationary_steps[agent]
                                
                                # Escalating penalty for being stuck
                                if stationary_steps <= 3:
                                    pass  # No penalty for brief stops
                                elif stationary_steps <= 10:
                                    shaped_reward -= 0.02
                                elif stationary_steps <= 30:
                                    shaped_reward -= 0.05
                                else:
                                    shaped_reward -= 0.08
                            else:
                                self._agent_stationary_steps[agent] = 0
                            
                            self._agent_last_positions[agent] = current_pos
                        
                        # --------------------------------------------------------
                        # REWARD 4: Deadlock penalty (moderate)
                        # --------------------------------------------------------
                        if info["deadlocks"][agent]:
                            self._agent_deadlock_steps[agent] = self._agent_deadlock_steps.get(agent, 0) + 1
                            deadlock_steps = self._agent_deadlock_steps[agent]
                            
                            # Escalating penalty for prolonged deadlock
                            if deadlock_steps <= 5:
                                shaped_reward -= 0.05
                            elif deadlock_steps <= 20:
                                shaped_reward -= 0.1
                            else:
                                shaped_reward -= 0.15
                        else:
                            self._agent_deadlock_steps[agent] = 0
                        
                        # --------------------------------------------------------
                        # REWARD 5: Time pressure (very small)
                        # --------------------------------------------------------
                        shaped_reward -= 0.002
                        
                        # --------------------------------------------------------
                        # REWARD 6: Team Completion Bonus
                        # --------------------------------------------------------
                        newly_completed = set()
                        for other_agent in self._env_handler.get_agents_handle():
                            if other_agent not in self._completed_agents:
                                other_obj = self._env_handler.env.agents[other_agent]
                                if other_obj.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                                    newly_completed.add(other_agent)
                        
                        if newly_completed:
                            shaped_reward += 0.1 * len(newly_completed)
                            self._completed_agents.update(newly_completed)
                        
                        # --------------------------------------------------------
                        # REWARD 7: Completion bonus (LARGE)
                        # --------------------------------------------------------
                        if done[agent]:
                            if agent_obj.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                                time_bonus = 1.0 - (count_steps / self._max_steps)
                                shaped_reward += 2.0 + 1.0 * time_bonus  # INCREASED
                                
                                # Clear tracking
                                if agent in self._prev_distances:
                                    del self._prev_distances[agent]
                                if agent in self._agent_deadlock_steps:
                                    del self._agent_deadlock_steps[agent]
                                if agent in self._agent_stationary_steps:
                                    del self._agent_stationary_steps[agent]
                        
                        # Clip reward
                        shaped_reward = np.clip(shaped_reward, -0.5, 3.0)
                        
                        # Store experience
                        if self._training and (update_values or done[agent]):
                            self._policy.step(
                                agent_prev_obs[agent],
                                agent_prev_action[agent],
                                shaped_reward,
                                agent_obs[agent],
                                done[agent],
                                agent
                            )
                            
                            agent_prev_obs[agent] = agent_obs[agent].copy()
                            agent_prev_action[agent] = action_dict[agent]

                        if next_obs[agent]:
                            agent_obs[agent] = self._obs_wrapper.normalize(next_obs[agent], self._env_handler, agent)

                        score += shaped_reward
                    
                    # Track stuck agents
                    for agent in self._env_handler.get_agents_handle():
                        agent_obj = self._env_handler.env.agents[agent]
                        
                        if info["deadlocks"][agent]:
                            deadlocked_agents.add(agent)
                        
                        if agent_obj.status == RailAgentStatus.ACTIVE and agent_obj.position is not None:
                            current_dist = abs(agent_obj.position[0] - agent_obj.target[0]) + \
                                           abs(agent_obj.position[1] - agent_obj.target[1])
                            
                            prev_dist = agent_last_distance.get(agent, current_dist)
                            
                            if current_dist >= prev_dist:
                                agent_no_progress_counter[agent] = agent_no_progress_counter.get(agent, 0) + 1
                            else:
                                agent_no_progress_counter[agent] = 0
                            
                            agent_last_distance[agent] = current_dist
                            
                            if agent_no_progress_counter.get(agent, 0) >= NO_PROGRESS_THRESHOLD:
                                stuck_agents.add(agent)

                    train_time = time.time() - train_time

                    log_data = {
                        "loss": self._policy.stats['loss'],
                        "time_act": act_time,
                        "time_train": train_time
                    }

                    self._logger.log_step(log_data, step)

                    if done['__all__']:
                        break

                # Collect statistics
                tasks_finished = 0
                tasks_failed = 0
                
                for idx in self._env_handler.get_agents_handle():
                    agent = self._env_handler.env.agents[idx]
                    
                    if agent.status == RailAgentStatus.DONE_REMOVED:
                        tasks_finished += 1
                    elif agent.status == RailAgentStatus.DONE:
                        if agent.position is None:
                            tasks_finished += 1
                        else:
                            tasks_failed += 1
                            stuck_agents.add(idx)
                    elif agent.status == RailAgentStatus.ACTIVE:
                        tasks_failed += 1
                        if idx not in deadlocked_agents:
                            stuck_agents.add(idx)
                    elif agent.status == RailAgentStatus.READY_TO_DEPART:
                        tasks_failed += 1

                n_agents = max(1, self._env_handler.env.get_num_agents())
                completion_rate = tasks_finished / n_agents
                
                all_stuck_agents = deadlocked_agents.union(stuck_agents)
                deadlock_rate = len(all_stuck_agents) / n_agents
                
                normalized_steps = count_steps / self._max_steps
                action_probs = action_count / np.sum(action_count)

                if episode_idx % 100 == 0:
                    print(f"\n=== Episode {episode_idx} Summary ===")
                    print(f"Steps: {count_steps}/{self._max_steps}")
                    print(f"Tasks finished: {tasks_finished}/{n_agents} = {completion_rate:.2%}")
                    print(f"Deadlocks: {len(deadlocked_agents)}, Stuck: {len(stuck_agents)}")
                    print(f"Exploration: {self._policy.stats.get('eps_val', 0):.3f}")
                    print(f"Action masking: DISABLED")
                    print(f"Staggered departures: {'ENABLED' if self._departure_controller else 'DISABLED'}")
                    if self._departure_controller:
                        print(f"  Spacing: {self._departure_controller.departure_spacing}")
                    print("=" * 40)

                self._logger.log_episode(
                    {
                        "completions": completion_rate,
                        "scores": score / (self._max_steps * n_agents),
                        "steps": normalized_steps,
                        "avg_delay": normalized_steps,
                        "loss": self._policy.stats["loss"] / np.sum(action_count)
                        if self._policy.stats["loss"] is not None
                        else None,
                        "deadlocks": deadlock_rate,
                        "exploration_prob": self._policy.stats.get("eps_val", 0.0),
                        "exploration_count": self._policy.stats.get("eps_counter", 0.0)
                        / np.sum(action_count),
                        **dict(zip(["act_" + str(i) for i in range(self._action_size)], action_probs)),
                    },
                    episode_idx,
                )
                
                eval_score = score
                self._policy.episode_end(self._env_handler.get_agents_handle())

                if episode_idx % self._trn_params['checkpoint_freq'] == 0:
                    end = "\n"
                    action_count = [1] * self._action_size

                    if self._training and self._save_checkpoints:
                        env_name = self._trn_params.get('env', 'unknown')
                        checkpoint_name = f"{str(self._policy)}-{env_name}-{episode_idx}.pth"
                        self._policy.save(os.path.join(self._sys_params['base_dir'], 'tmp', 'checkpoints',
                                                    checkpoint_name),
                                        overwrite=True)
                        self._policy.save_best()
                else:
                    end = " "

                if self._verbose:
                    self._env_handler.print_results(episode_idx, self._logger.get_window('scores'),
                                                    self._logger.get_window('completions'), 
                                                    self._logger.get_window('avg_delay'),
                                                    self._logger.get_window('deadlocks'), 
                                                    action_probs, end)

            self._logger.run_end(self._trn_params, 
                                 eval_score / (self._max_steps * self._env_handler.env.get_num_agents()),
                                 run_id)

        return time.time() - start_time


class EnvHandler:
    def __init__(self, env_filename, obs_builder, rendering=False):
        with open(env_filename) as json_file:
            self._full_env_params = json.load(json_file)

        self._obs_builder = obs_builder
        self._rendering = rendering
        self.deadlocks_detector = DeadlocksDetector()

    def update(self, env="r1.s", seed=None):
        self._params = self._full_env_params[env]

        self.x_dim = self._params['x_dim']
        self.y_dim = self._params['y_dim']
        self.n_cities = self._params['n_cities']
        self.grid_mode = self._params['grid_mode']
        self.max_rails_between_cities = self._params['max_rails_between_cities']
        self.max_rails_in_city = self._params['max_rails_in_city']
        self.n_agents = self._params['n_agents']
        min_mal, max_mal = self._params["malfunction_duration"]

        if "malfunction_rate" in self._params:
             malfunction_rate = self._params["malfunction_rate"]
        elif "min_malfunction_interval" in self._params and self._params["min_malfunction_interval"] > 0:
             malfunction_rate = 1.0 / self._params["min_malfunction_interval"]
        else:
             malfunction_rate = 0.0

        self.mal_params = mal_gen.MalfunctionParameters(malfunction_rate, min_mal, max_mal)
        
        try:
            self.env = RailEnv(
                width=self.x_dim,
                height=self.y_dim,
                rail_generator=sparse_rail_generator(
                    max_num_cities=self.n_cities,
                    seed=seed,
                    grid_mode=self.grid_mode,
                    max_rails_between_cities=self.max_rails_between_cities,
                    max_rails_in_city=self.max_rails_in_city
                ),
                schedule_generator=sparse_schedule_generator(),
                number_of_agents=self.n_agents,
                obs_builder_object=self._obs_builder,
                malfunction_generator=mal_gen.ParamMalfunctionGen(self.mal_params),
                close_following=False,
                random_seed=seed
            )
        except AttributeError:
            self.env = RailEnv(
                width=self.x_dim,
                height=self.y_dim,
                rail_generator=sparse_rail_generator(
                    max_num_cities=self.n_cities,
                    seed=seed,
                    grid_mode=self.grid_mode,
                    max_rails_between_cities=self.max_rails_between_cities,
                    max_rails_in_city=self.max_rails_in_city
                ),
                schedule_generator=sparse_schedule_generator(),
                number_of_agents=self.n_agents,
                obs_builder_object=self._obs_builder,
                malfunction_generator_and_process_data=mal_gen.malfunction_from_params(self.mal_params),
                random_seed=seed
            )

        if self._rendering:
            self._renderer = RenderTool(self.env)

    def print_results(self, episode_idx, scores_window, completion_window, delay_window, deadlock_window, action_probs, end):
        print(
            '\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tCompletion Rate: {:.2f}%\t '
            'Deadlock Rate: {:.2f}%\t Avg Delay: {:.2f}%\t Action Probabilities: \t {}'.format(
                self.env.get_num_agents(),
                self._params['x_dim'], self._params['y_dim'],
                episode_idx,
                np.mean(scores_window),
                100 * np.mean(completion_window),
                100 * np.mean(deadlock_window),
                100 * np.mean(delay_window),
                action_probs,
            ), end=end)

    def step(self, action_dict):
        next_obs, all_rewards, done, info = self.env.step(action_dict)

        deadlocks = self.deadlocks_detector.step(self.env)
        info["deadlocks"] = {}
        for agent in self.get_agents_handle():
            info["deadlocks"][agent] = deadlocks[agent]

        if self._rendering:
            self._renderer.render_env(show=True, show_observations=False, show_predictions=False)

        return next_obs, all_rewards, done, info

    def get_num_agents(self):
        return self.n_agents

    def get_agents_handle(self):
        return self.env.get_agent_handles()

    def reset(self):
        obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)

        self.deadlocks_detector.reset(self.env.get_num_agents())
        info["deadlocks"] = {}

        for agent in self.get_agents_handle():
            info["deadlocks"][agent] = self.deadlocks_detector.deadlocks[agent]

        if self._rendering:
            self._renderer.reset()

        return obs, info