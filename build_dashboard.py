"""build_dashboard.py : assembles dashboard_template.html + everything
recorded by record_for_dashboard.py and make_figures_generic.py + the
comparison numbers from compare_baselines_generic.py into one
self-contained HTML file you can double-click and open in any browser.

Run this LAST, after:
    1. train_generic.py
    2. compare_baselines_generic.py
    3. make_figures_generic.py
    4. record_for_dashboard.py

Usage:
    python3 build_dashboard.py
Output:
    dashboard.html   <-- open this in your browser / present it live
"""
import base64
import json
import os


def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def load_json(path):
    with open(path) as f:
        return json.load(f)


required = [
    'screens/osm_network.png', 'screens/osm_reward_curve.png',
    'screens/osm_comparison.png', 'screens/osm_cv_detection.png',
    'logs/replay_fixed.json', 'logs/replay_dqn.json',
    'logs/osm_comparison_summary.json', 'logs/road_lines.json',
    'logs/explain_dqn.json', 'logs/tls_meta.json', 'dashboard_template.html',
]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    print('Missing files -- run the earlier scripts first:')
    for m in missing:
        print('  -', m)
    raise SystemExit(1)

imgs = {
    'network': b64('screens/osm_network.png'),
    'reward': b64('screens/osm_reward_curve.png'),
    'compare': b64('screens/osm_comparison.png'),
    'cv': b64('screens/osm_cv_detection.png'),
}
fixed = load_json('logs/replay_fixed.json')
dqn = load_json('logs/replay_dqn.json')
summary = load_json('logs/osm_comparison_summary.json')
road_lines = load_json('logs/road_lines.json')
explain_data = load_json('logs/explain_dqn.json')
green_phases = load_json('logs/tls_meta.json')['green_phases']

data = {'fixed': fixed, 'dqn': dqn, 'summary': summary}

with open('dashboard_template.html', encoding='utf-8') as f:
    html = f.read()

html = html.replace('__IMG_OSM_NETWORK__', imgs['network'])
html = html.replace('__IMG_OSM_REWARD__', imgs['reward'])
html = html.replace('__IMG_OSM_COMPARE__', imgs['compare'])
html = html.replace('__IMG_OSM_CV__', imgs['cv'])
html = html.replace('__DATA_JSON__', json.dumps(data))
html = html.replace('__ROAD_LINES_JSON__', json.dumps(road_lines))
html = html.replace('__GREEN_PHASES_JSON__', json.dumps(green_phases))
html = html.replace('__EXPLAIN_DATA_JSON__', json.dumps(explain_data))

assert '__IMG' not in html and '__DATA' not in html and '__ROAD' not in html \
    and '__GREEN' not in html and '__EXPLAIN' not in html, \
    'Template placeholder left unfilled -- check dashboard_template.html matches this script'

with open('dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Wrote dashboard.html ({len(html) / 1e6:.1f} MB) -- open it in your browser.')
