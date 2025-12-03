from abc import ABC, abstractmethod
import os
import itertools
import json
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class Logger(ABC):
    """
    Base logger interface.

    In this version we are no longer tied to TensorFlow, TensorBoard or wandb.
    We only keep the generic logging structure.
    """

    def __init__(self, base_dir: str, parameters: Dict[str, Any],
                 tuning: bool = False, sync: bool = False) -> None:
        """
        `parameters` is expected to contain at least:

          - 'attributes': dict describing which kind of log each metric has
                          (e.g. {"reward": ["val", "avg"]})
          - 'log_dir'   : log directory relative to `base_dir`
          - (optional when tuning):
                'hp_dir', 'hp_params_filename'
        """
        self._attributes: Dict[str, List[str]] = parameters.get("attributes", {})
        self._base_dir: str = base_dir
        self._log_dir: str = parameters.get("log_dir", "logs")
        self._hp_tuning: bool = tuning
        self._sync: bool = sync

        # Will be set in `run_start`
        self._run_dir: Optional[str] = None

        if self._hp_tuning:
            self._hp_dir: str = parameters.get("hp_dir", "hparams")
            self._hp_params_filename: str = parameters.get("hp_params_filename", "")
            self._init_hp()
        else:
            self._hparams: List[Dict[str, Any]] = []
            self._parameter_list: Dict[str, List[Any]] = {}
            self._combinations: List[Dict[str, Any]] = []

    # ---------- Paths ----------

    def get_log_dir(self) -> str:
        """
        Root log directory (without driver), e.g.:

        base_dir + log_dir
        """
        return os.path.join(self._base_dir, self._log_dir)

    def get_hp_run_dir(self) -> str:
        """
        Log directory for hparams of a single run.
        """
        return os.path.join(
            self._base_dir,
            self._log_dir,
            self._get_driver_dir(),
            self._hp_dir,
            self._run_dir or "",
        )

    def get_hp_dir(self) -> str:
        """
        Global hparams directory for all runs.
        """
        return os.path.join(
            self._base_dir,
            self._log_dir,
            self._get_driver_dir(),
            self._hp_dir,
        )

    def get_run_dir(self) -> str:
        """
        Log directory of the current run.
        """
        return os.path.join(
            self._base_dir,
            self._log_dir,
            self._get_driver_dir(),
            self._run_dir or "",
        )

    # ---------- Hyperparameters ----------

    def get_run_params(self) -> List[Dict[str, Any]]:
        """
        If tuning is enabled, return the list of hparam combinations.
        Otherwise, return just `[{}]` (i.e. "no hparams").
        """
        if self._hp_tuning:
            return self._combinations
        return [{}]

    # ---------- Logging ----------

    def log_step(self, pack: Dict[str, Any], idx: int) -> None:
        """
        Log metrics for a single step.

        pack: dict of metrics, e.g. {"reward": 1.2, "delay": 5.0}
        idx:  step index
        """
        self._log(pack, "step", idx)

    def log_episode(self, pack: Dict[str, Any], idx: int) -> None:
        """
        Log metrics for a single episode.
        """
        self._log(pack, "epsd", idx)

    def get_window(self, key: str) -> Optional[deque]:
        """
        Get the moving-average window for a metric, if it exists.
        """
        return getattr(self, "_windows", {}).get(key, None)

    # ---------- Abstract methods ----------

    @abstractmethod
    def run_start(self, run_params: Dict[str, Any], agent_name: str) -> None:
        """
        Called at the beginning of a new run (before the training/eval loop).
        """
        pass

    @abstractmethod
    def run_end(self, params: Dict[str, Any],
                scores: Optional[float],
                episode_idx: int) -> None:
        """
        Called at the end of a run (e.g. to save a summary of the results).
        """
        pass

    @abstractmethod
    def _log(self, pack: Dict[str, Any], type: str, idx: int) -> None:
        """
        Internal logging method.

        `type` can be "step" or "epsd".
        """
        pass

    @abstractmethod
    def _init_hp(self) -> None:
        """
        Prepare hyperparameter combinations when tuning is enabled.
        """
        pass

    @abstractmethod
    def _get_driver_dir(self) -> str:
        """
        Subdirectory used by the logger (e.g. "tensorboard" or "wandb").
        """
        pass


class TensorboardLogger(Logger):
    """
    Simple logger that writes logs to CSV files instead of TensorBoard.
    """

    def __init__(self, base_dir: str, parameters: Dict[str, Any],
                 tuning: bool = False, sync: bool = False) -> None:
        super().__init__(base_dir, parameters, tuning=tuning, sync=sync)
        self._windows: Dict[str, deque] = {}
        self._step_log_file: Optional[str] = None
        self._epsd_log_file: Optional[str] = None

    def run_start(self, run_params: Dict[str, Any], agent_name: str) -> None:
        # Run directory name
        self._run_dir = agent_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create run directory
        os.makedirs(self.get_run_dir(), exist_ok=True)

        # Prepare moving-average windows
        self._windows = {}
        for attr, kinds in self._attributes.items():
            if "avg" in kinds:
                self._windows[attr] = deque(maxlen=100)

        # CSV files for step and episode logs
        self._step_log_file = os.path.join(self.get_run_dir(), "steps.csv")
        self._epsd_log_file = os.path.join(self.get_run_dir(), "episodes.csv")

        # Write headers
        self._write_csv_header(self._step_log_file)
        self._write_csv_header(self._epsd_log_file)

    def run_end(self, params: Dict[str, Any],
                scores: Optional[float],
                run_idx: int) -> None:
        """
        Save a JSON summary of the run.
        """
        summary_path = os.path.join(self.get_run_dir(), "summary.json")
        data = {
            "run_index": run_idx,
            "scores": float(scores) if scores is not None else None,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _write_csv_header(self, path: str) -> None:
        """
        Write CSV header line:
        columns: idx, type, metrics (val/avg)
        """
        cols = ["idx", "type"]
        for attr, kinds in self._attributes.items():
            if "val" in kinds:
                cols.append(attr + "_val")
            if "avg" in kinds:
                cols.append(attr + "_avg")

        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")

    def _log(self, pack: Dict[str, Any], type: str, idx: int) -> None:
        """
        Log to CSV:
        - if type == "step" → steps.csv
        - if type == "epsd" → episodes.csv
        """
        if type == "step":
            path = self._step_log_file
        else:
            path = self._epsd_log_file

        if path is None:
            return

        row: List[Any] = [idx, type]
        for attr, kinds in self._attributes.items():
            val_to_write_val: Any = ""
            val_to_write_avg: Any = ""

            val = pack.get(attr, None)
            if val is not None:
                # Instantaneous value
                if "val" in kinds:
                    val_to_write_val = float(val)
                # Moving average
                if "avg" in kinds:
                    self._windows[attr].append(val)
                    val_to_write_avg = float(np.mean(self._windows[attr]))

            # Order: first val then avg (if defined in attributes)
            if "val" in kinds:
                row.append(val_to_write_val)
            if "avg" in kinds:
                row.append(val_to_write_avg)

        with open(path, "a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")

    def _init_hp(self) -> None:
        """
        Build hyperparameter combinations from a JSON file,
        without using TensorBoard hparams.
        """
        self._hparams = []
        self._parameter_list = {}
        self._combinations = []

        if not getattr(self, "_hp_params_filename", None):
            return

        hp_path = os.path.join(self._base_dir, self._hp_params_filename)
        if not os.path.exists(hp_path):
            return

        with open(hp_path, "r", encoding="utf-8") as json_file:
            hyper_params = json.load(json_file)
            for key, descr in hyper_params.items():
                if descr["type"] == "discrete":
                    values = descr["values"]
                elif descr["type"] == "interval_real":
                    values = list(
                        np.linspace(descr["min"], descr["max"], descr["n"])
                    )
                elif descr["type"] == "interval_int":
                    values = list(
                        range(descr["min"], descr["max"], descr["step"])
                    )
                else:
                    values = []

                self._parameter_list[key] = values

            # All combinations
            self._combinations = [
                dict(zip(self._parameter_list.keys(), combo))
                for combo in itertools.product(*self._parameter_list.values())
            ]

    def _get_driver_dir(self) -> str:
        return "tensorboard"


class WandBLogger(TensorboardLogger):
    """
    This version no longer actually uses wandb.
    It is only kept for API compatibility with the original code and
    behaves exactly like TensorboardLogger (i.e. a CSV logger),
    but under a different subdirectory.
    """

    def __init__(self, base_dir: str, parameters: Dict[str, Any],
                 tuning: bool, sync: bool = False) -> None:
        super().__init__(base_dir, parameters, tuning=tuning, sync=sync)

    def _get_driver_dir(self) -> str:
        return "wandb"


class WandBAndTensorboardLogger(TensorboardLogger):
    """
    Same idea as above: this class exists because the original code might
    call it. In practice it behaves exactly like TensorboardLogger, but
    uses another directory name.
    """

    def __init__(self, base_dir: str, parameters: Dict[str, Any],
                 tuning: bool, sync: bool = False) -> None:
        super().__init__(base_dir, parameters, tuning=tuning, sync=sync)

    def _get_driver_dir(self) -> str:
        return "wandb_tensorboard"
