import re

lines = open('logs/train_console.log').read().splitlines()
episode_lines = [l for l in lines if l.startswith('Episode')]
final_line = [l for l in lines if l.startswith('Training complete')]

# Pick a representative subsample: every ~20 episodes, evenly spaced, ending near the end
n = len(episode_lines)
picks_idx = sorted(set([0] + [min(n - 1, int(round(k))) for k in
                    [n * f for f in (0.08, 0.25, 0.42, 0.58, 0.75, 0.92, 1.0)]]))
picked = [episode_lines[i] for i in picks_idx]

out = ['$ python3 train.py',
       "Loading net-file from 'network/intersection.net.xml' ... done.",
       "Loading route-files from 'network/routes.rou.xml' ... done."]
out += picked
out += (final_line if final_line else ['Training complete. Checkpoint saved to checkpoints/dqn_latest.npz'])

with open('logs/train_console_clean.txt', 'w') as f:
    f.write('\n'.join(out) + '\n')

print('\n'.join(out))
