import subprocess, json, sys

CURRICULUM = [
    ("phase0_easy",   100),
    ("phase1_light_malfunction", 150),
    ("phase2_medium", 200),
    ("phase3_hard",   300),
]

AGENTS = ["DQNAgent", "PPOAgent"]

def run(agent, env, episodes):
    print(f"\n=== Running {agent} on {env} for {episodes} episodes ===")

    with open("parameters/setup.json", "r") as f:
        cfg = json.load(f)
    cfg["sys"]["agent_class"] = agent
    cfg["trn"]["env"] = env

    # overwrite setup.json
    with open("parameters/setup.json", "w") as f:
        json.dump(cfg, f, indent=4)

    cmd = [
        sys.executable, "main.py",
        "--training", "best",
        "--episodes", str(episodes),
        "--verbose"
    ]
    subprocess.run(cmd)

def main():
    for agent in AGENTS:
        print(f"\n=============================\n Curriculum for {agent} \n=============================")
        for env, episodes in CURRICULUM:
            run(agent, env, episodes)

if __name__ == "__main__":
    main()
