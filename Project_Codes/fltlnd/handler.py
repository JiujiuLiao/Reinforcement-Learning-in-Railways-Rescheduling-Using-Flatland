import os
from typing import Optional
from fltlnd.utils import TrainingMode
import json
import time
import random
from flatland.envs import malfunction_generators as mal_gen
import numpy as np

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


class ExcHandler:
    def __init__(self, params: dict, training_mode: TrainingMode, rendering: bool, checkpoint: Optional[str],
                 synclog: bool, verbose: bool):
        self._sys_params = params['sys']  # System
        self._obs_params = params['obs']  # Observation
        self._trn_params = params['trn']  # Training
        self._log_params = params['log']  # Policy

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
        # For PPO use AgentEpisodeBuffer; for other agents use the class specified in setup.json
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
        # Give observation wrapper access to env handler
        self._obs_wrapper.env = self._env_handler

        # The action space of flatland is 5 discrete actions
        self._action_size = 5
        self._state_size = self._obs_wrapper.get_state_size() + self._obs_wrapper.n_global_features # Additional features for enhanced observation

        self._logger = self._logger_class(self._sys_params['base_dir'], self._log_params, self._tuning, synclog)

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

            # Max number of steps per episode
            self._max_steps = int(4 * 2 * (self._env_handler._params['x_dim'] + self._env_handler._params['y_dim'] + (
                    self._env_handler.get_num_agents() / self._env_handler._params['n_cities'])))

            eval_score = None
            for episode_idx in range(n_episodes):
                self._policy.episode_start()

                score = 0
                action_dict = dict()
                action_count = [0] * self._action_size
                agent_obs = [None] * self._env_handler.get_num_agents()
                agent_prev_obs = [None] * self._env_handler.get_num_agents()
                agent_prev_action = [2] * self._env_handler.get_num_agents()
                agent_prev_rewards = [0] * self._env_handler.get_num_agents()
                agent_prev_done = [0] * self._env_handler.get_num_agents()
                update_values = False

                # Reset environment
                obs, info = self._env_handler.reset()
                
                # ============================================================
                # FIXED: Tracking for deadlock and stuck detection
                # ============================================================
                deadlocked_agents = set()          # Agents detected as deadlocked (hard deadlock)
                stuck_agents = set()               # Agents stuck in loops (soft deadlock)
                agent_last_distance = {}           # Track distance to target for each agent
                agent_no_progress_counter = {}     # Count steps without making progress
                NO_PROGRESS_THRESHOLD = 50         # Steps without progress to consider "stuck"
                
                # Reset per-episode tracking for reward shaping
                self._prev_distances = {}
                self._agent_deadlock_steps = {}  # Track how long each agent has been in deadlock
                
                # Get number of agents for reward normalization
                num_agents = self._env_handler.get_num_agents()

                # Build agent-specific observations
                for agent in self._env_handler.get_agents_handle():
                    if obs[agent]:
                        agent_obs[agent] = self._obs_wrapper.normalize(
                            obs[agent],
                            self._env_handler,
                            agent
                        )
                        agent_prev_obs[agent] = agent_obs[agent].copy()

                count_steps = 0
                # Run episode 
                for step in range(self._max_steps - 1):
                    count_steps += 1

                    act_time = time.time()
                    for agent in self._env_handler.get_agents_handle():
                        agent_obj = self._env_handler.env.agents[agent]
                        
                        if info['action_required'][agent]:
                            update_values = True
                            action = self._policy.act(agent_obs[agent])
                            action_count[action] += 1
                        else: 
                            update_values = False
                            # FIX: Check if agent is stopped and try to restart
                            # RailAgentStatus.ACTIVE = 1, and if position hasn't changed, agent might be stopped
                            if agent_obj.status == RailAgentStatus.ACTIVE:
                                # Agent is active but no action required - either on straight track or stopped
                                # Send FORWARD to ensure movement continues
                                action = 2  # MOVE_FORWARD - this will restart stopped agents
                            else:
                                action = 0  # DO_NOTHING for non-active agents
                        
                        action_dict.update({agent: action})
                    act_time = time.time() - act_time

                    # Environment step
                    next_obs, all_rewards, done, info = self._env_handler.step(action_dict)

                    # ============================================================
                    # REWARD SHAPING 
                    # ============================================================
                    train_time = time.time()
                    for agent in self._env_handler.get_agents_handle():
                        shaped_reward = 0.0
                        agent_obj = self._env_handler.env.agents[agent]
                        
                        # ----------------------------------------------------------
                        # FIX 1: Use normalized, bounded rewards
                        # All rewards are scaled to roughly [-1, +1] range
                        # ----------------------------------------------------------
                        
                        # 1. Progress reward (dense signal)
                        #    Normalized by approximate map diagonal
                        if agent_obj.position is not None and agent_obj.target is not None:
                            current_dist = abs(agent_obj.position[0] - agent_obj.target[0]) + \
                                          abs(agent_obj.position[1] - agent_obj.target[1])
                            
                            # Normalize by map size
                            map_diagonal = self._env_handler.x_dim + self._env_handler.y_dim
                            
                            prev_dist = self._prev_distances.get(agent, current_dist)
                            
                            # Progress is change in distance, normalized
                            progress = (prev_dist - current_dist) / map_diagonal
                            shaped_reward += progress * 0.5  # Scale factor
                            
                            self._prev_distances[agent] = current_dist
                        
                        # 2. Completion reward (sparse but important)
                        #    FIX: Use moderate, fixed bonus regardless of agent count
                        if done[agent]:
                            if agent_obj.status == RailAgentStatus.DONE:
                                # Successful completion - moderate bonus
                                shaped_reward += 1.0
                                # Clear tracking
                                if agent in self._prev_distances:
                                    del self._prev_distances[agent]
                                if agent in self._agent_deadlock_steps:
                                    del self._agent_deadlock_steps[agent]
                            else:
                                # Failed to reach target - small penalty
                                shaped_reward -= 0.1
                        
                        # 3. Deadlock handling
                        #    FIX: Use decaying penalty, not constant large penalty
                        #    This prevents the "just stop moving" exploit
                        if info["deadlocks"][agent]:
                            # Track how many steps this agent has been deadlocked
                            self._agent_deadlock_steps[agent] = self._agent_deadlock_steps.get(agent, 0) + 1
                            deadlock_steps = self._agent_deadlock_steps[agent]
                            
                            # Penalty increases over time but is bounded
                            # First few steps: small penalty (might resolve naturally)
                            # Longer deadlock: larger penalty (but capped)
                            if deadlock_steps <= 5:
                                shaped_reward -= 0.05
                            elif deadlock_steps <= 20:
                                shaped_reward -= 0.1
                            else:
                                shaped_reward -= 0.2  # Cap the penalty
                        else:
                            # Reset deadlock counter if agent is not deadlocked
                            self._agent_deadlock_steps[agent] = 0
                        
                        # 4. Time pressure (encourage efficiency)
                        #    FIX: Very small penalty, normalized by max steps
                        shaped_reward -= 0.001
                        
                        # 5. Action-specific shaping
                        #    FIX: Remove STOP penalty - it causes reward hacking!
                        #    Instead, only penalize illegal actions
                        if not info['action_required'][agent] and action_dict[agent] != 0:
                            shaped_reward -= 0.01  # Small penalty for unnecessary action
                        
                        # ----------------------------------------------------------
                        # FIX 2: Clip total reward to prevent extreme values
                        # ----------------------------------------------------------
                        shaped_reward = np.clip(shaped_reward, -2.0, 2.0)
                        
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
                            agent_obs[agent] = self._obs_wrapper.normalize(
                                next_obs[agent],
                                self._env_handler,
                                agent
                            )

                        score += shaped_reward
                    
                    # ============================================================
                    # FIXED: Track deadlocked AND stuck agents
                    # ============================================================
                    for agent in self._env_handler.get_agents_handle():
                        agent_obj = self._env_handler.env.agents[agent]
                        
                        # 1. Hard deadlock from detector
                        if info["deadlocks"][agent]:
                            deadlocked_agents.add(agent)
                        
                        # 2. Soft deadlock: agent not making progress toward target
                        if agent_obj.status == RailAgentStatus.ACTIVE and agent_obj.position is not None:
                            current_dist = abs(agent_obj.position[0] - agent_obj.target[0]) + \
                                           abs(agent_obj.position[1] - agent_obj.target[1])
                            
                            prev_dist = agent_last_distance.get(agent, current_dist)
                            
                            # Check if agent is making progress (getting closer to target)
                            if current_dist >= prev_dist:
                                # Not making progress
                                agent_no_progress_counter[agent] = agent_no_progress_counter.get(agent, 0) + 1
                            else:
                                # Making progress - reset counter
                                agent_no_progress_counter[agent] = 0
                            
                            agent_last_distance[agent] = current_dist
                            
                            # If no progress for too long, mark as stuck
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

                # ============================================================
                # FIXED: Collect training statistics correctly
                # ============================================================
                tasks_finished = 0
                tasks_failed = 0
                
                for idx in self._env_handler.get_agents_handle():
                    agent = self._env_handler.env.agents[idx]
                    
                    # Check if agent successfully completed
                    if agent.status == RailAgentStatus.DONE_REMOVED:
                        tasks_finished += 1
                    elif agent.status == RailAgentStatus.DONE:
                        # DONE with position=None means reached target and was removed
                        if agent.position is None:
                            tasks_finished += 1
                        else:
                            # DONE but still has position - did not reach target
                            tasks_failed += 1
                            stuck_agents.add(idx)
                    elif agent.status == RailAgentStatus.ACTIVE:
                        # Agent still ACTIVE at episode end = failed to complete
                        tasks_failed += 1
                        # If not already marked as deadlocked, mark as stuck
                        if idx not in deadlocked_agents:
                            stuck_agents.add(idx)
                    elif agent.status == RailAgentStatus.READY_TO_DEPART:
                        # Agent never departed - count as failed
                        tasks_failed += 1

                n_agents = max(1, self._env_handler.env.get_num_agents())
                completion_rate = tasks_finished / n_agents
                
                # Combine hard deadlocks and soft stuck agents for total "failed due to stuck" rate
                all_stuck_agents = deadlocked_agents.union(stuck_agents)
                deadlock_rate = len(all_stuck_agents) / n_agents
                
                normalized_steps = count_steps / self._max_steps
                action_probs = action_count / np.sum(action_count)

                # Diagnostic - remove after verification
                print(f"\n=== Episode {episode_idx} Summary ===")
                print(f"Steps: {count_steps}/{self._max_steps}")
                print(f"Episode ended: {'all done' if done['__all__'] else 'max steps reached'}")
                print(f"Tasks finished: {tasks_finished}/{n_agents} = {completion_rate:.2%}")
                print(f"Hard deadlocks: {deadlocked_agents}")
                print(f"Soft stuck (no progress): {stuck_agents}")
                print(f"Total stuck/deadlock rate: {deadlock_rate:.2%}")
                for idx in self._env_handler.get_agents_handle():
                    agent = self._env_handler.env.agents[idx]
                    status_name = {0: "READY_TO_DEPART", 1: "ACTIVE", 2: "DONE", 3: "DONE_REMOVED"}.get(agent.status, "UNKNOWN")
                    print(f"  Agent {idx}: status={status_name}({agent.status}), position={agent.position}, target={agent.target}")
                print("=" * 40)

                self._logger.log_episode(
                    {
                        **{
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
                        },
                        **dict(
                            zip(
                                ["act_" + str(i) for i in range(self._action_size)],
                                action_probs,
                            )
                        ),
                    },
                    episode_idx,
                )
                
                eval_score = score
                self._policy.episode_end(self._env_handler.get_agents_handle())

                if episode_idx % self._trn_params['checkpoint_freq'] == 0:
                    end = "\n"
                    action_count = [1] * self._action_size

                    if self._training and self._save_checkpoints:
                        self._policy.save(os.path.join(self._sys_params['base_dir'], 'tmp', 'checkpoints',
                                                       str(self._policy) + '-' + str(episode_idx) + '.pth'),
                                          overwrite=True)
                        self._policy.save_best()

                else:
                    end = " "

                if self._verbose:
                    self._env_handler.print_results(episode_idx, self._logger.get_window('scores'),
                                                    self._logger.get_window('completions'), self._logger.get_window('avg_delay'),self._logger.get_window('deadlocks'), action_probs, end)

            self._logger.run_end(self._trn_params, eval_score / (self._max_steps * self._env_handler.env.get_num_agents()),
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

        self.mal_params = mal_gen.MalfunctionParameters(
            malfunction_rate,
            min_mal,
            max_mal
        )
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