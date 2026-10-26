import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('logs/train_console.log') as f:
    lines = [l.rstrip('\n') for l in f if l.strip() and 'Retrying' not in l and 'Warning' not in l]

# keep header + episode lines + final line, trimmed to fit a readable terminal panel
show_lines = ['$ python3 train.py'] + lines[:26]
if len(lines) > 26:
    show_lines.append('...')
    show_lines += lines[-3:]

fig_h = 0.28 * len(show_lines) + 0.6
fig, ax = plt.subplots(figsize=(9.5, fig_h))
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')
ax.axis('off')
y = 1.0
dy = 1.0 / (len(show_lines) + 1)
for line in show_lines:
    color = '#58a6ff' if line.startswith('$') else ('#7ee787' if 'complete' in line.lower() else '#c9d1d9')
    ax.text(0.01, y, line, family='monospace', fontsize=9.3, color=color,
            va='top', ha='left', transform=ax.transAxes)
    y -= dy

plt.tight_layout(pad=0.6)
plt.savefig('screens/fig_11_4_console.png', dpi=150, facecolor=fig.get_facecolor())
print('saved console screenshot,', len(show_lines), 'lines')
