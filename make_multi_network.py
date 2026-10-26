"""make_multi_network.py : builds a multi-intersection SUMO scenario.

Generates an N x M grid of signalised junctions with SUMO's own
netgenerate (same lane count / speed / static TLS type as the single
synthetic intersection in network/), then writes a time-varying demand
file in the same style as network/routes.rou.xml: off-peak -> peak with
an asymmetric corridor -> off-peak tail. Vehicles cross several
junctions, so neighbouring signals genuinely affect each other.

Usage:
    python3 make_multi_network.py                 # 2x2 grid (4 signals)
    python3 make_multi_network.py --nx 3 --ny 1   # 3-signal arterial
    python3 make_multi_network.py --nx 3 --ny 3   # 3x3 grid (9 signals)

Output: network_multi/grid.net.xml, grid.rou.xml, grid.sumocfg
"""
import argparse
import os
import random
import subprocess
import sys

sys.path.append(os.path.join(os.environ.get('SUMO_HOME', '/usr/share/sumo'), 'tools'))
import sumolib  # noqa: E402

OUT_DIR = 'network_multi'


def build_net(nx, ny, length, out_net):
    netgenerate = sumolib.checkBinary('netgenerate')
    cmd = [netgenerate, '--grid',
           '--grid.x-number', str(nx), '--grid.y-number', str(ny),
           '--grid.length', str(length), '--grid.attach-length', str(length),
           '--default.lanenumber', '2', '--default.speed', '13.9',
           '--tls.guess', 'true', '--tls.default-type', 'static',
           '--no-turnarounds', 'true', '-o', out_net]
    subprocess.run(cmd, check=True)


def fringe_edges(net):
    """Edges entering the network from a dead-end node (sources) and
    edges leaving the network to a dead-end node (sinks)."""
    sources, sinks = [], []
    for e in net.getEdges():
        if e.getFunction() == 'internal':
            continue
        if e.getFromNode().getType() == 'dead_end':
            sources.append(e)
        if e.getToNode().getType() == 'dead_end':
            sinks.append(e)
    return sources, sinks


def side_of(edge, bbox, dead_end_node):
    (xmin, ymin), (xmax, ymax) = bbox
    x, y = dead_end_node.getCoord()
    d = {'W': abs(x - xmin), 'E': abs(x - xmax), 'S': abs(y - ymin), 'N': abs(y - ymax)}
    return min(d, key=d.get)


def write_routes(net, out_rou, seed=42):
    rng = random.Random(seed)
    sources, sinks = fringe_edges(net)
    bbox = net.getBBoxXY()
    src_side = {e.getID(): side_of(e, bbox, e.getFromNode()) for e in sources}
    snk_side = {e.getID(): side_of(e, bbox, e.getToNode()) for e in sinks}
    opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    # The "main corridor" for the peak: the first N->S column and the
    # first W->E row get heavy demand, everything else stays light, so
    # different junctions need different timing (fixed-time can't adapt).
    def xy(src):
        return net.getEdge(src).getFromNode().getCoord()
    ns_sources = [s for s, side in src_side.items() if side in ('N', 'S')]
    ew_sources = [s for s, side in src_side.items() if side in ('E', 'W')]
    min_x = min(xy(s)[0] for s in ns_sources)
    min_y = min(xy(s)[1] for s in ew_sources)
    # both directions of the left-most column and of the bottom-most row
    main_ns = {s for s in ns_sources if abs(xy(s)[0] - min_x) < 1.0}
    main_ew = {s for s in ew_sources if abs(xy(s)[1] - min_y) < 1.0}

    def straight_sink(src):
        """Sink on the opposite side that is closest in line with the source."""
        se = net.getEdge(src)
        sx, sy = se.getFromNode().getCoord()
        want = opposite[src_side[src]]
        cands = [k for k, side in snk_side.items() if side == want]
        if want in ('N', 'S'):
            return min(cands, key=lambda k: abs(net.getEdge(k).getToNode().getCoord()[0] - sx))
        return min(cands, key=lambda k: abs(net.getEdge(k).getToNode().getCoord()[1] - sy))

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<routes>',
             '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" '
             'minGap="2.5" maxSpeed="16.6"/>', '']
    fid = 0
    periods = [  # (begin, end, through_main, through_other, turning)
        (0, 600, 0.07, 0.05, 0.015),     # off-peak
        (600, 1800, 0.20, 0.04, 0.02),   # peak, heavy on main corridor
        (1800, 2400, 0.06, 0.05, 0.015), # off-peak tail
    ]
    for begin, end, p_main, p_other, p_turn in periods:
        lines.append(f'    <!-- {begin}-{end}s -->')
        for src in sorted(src_side):
            dst = straight_sink(src)
            is_main = src in main_ns or src in main_ew
            p = p_main if is_main else p_other
            fid += 1
            lines.append(f'    <flow id="f{fid}" type="car" from="{src}" to="{dst}" '
                         f'begin="{begin}" end="{end}" probability="{p:.3f}" '
                         f'departLane="best" departSpeed="max"/>')
            # a couple of turning / cross-network movements per source
            others = [k for k in snk_side if k != dst and snk_side[k] != src_side[src]]
            for dst2 in rng.sample(others, min(2, len(others))):
                fid += 1
                lines.append(f'    <flow id="f{fid}" type="car" from="{src}" to="{dst2}" '
                             f'begin="{begin}" end="{end}" probability="{p_turn:.3f}" '
                             f'departLane="best" departSpeed="max"/>')
        lines.append('')
    lines.append('</routes>')
    with open(out_rou, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return fid


def write_cfg(out_cfg, net_name, rou_name):
    with open(out_cfg, 'w') as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_name}"/>
        <route-files value="{rou_name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="2400"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
    </processing>
</configuration>
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=2, help='junctions along x')
    ap.add_argument('--ny', type=int, default=2, help='junctions along y')
    ap.add_argument('--length', type=float, default=200.0, help='block length (m)')
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    net_path = os.path.join(args.out, 'grid.net.xml')
    rou_path = os.path.join(args.out, 'grid.rou.xml')
    cfg_path = os.path.join(args.out, 'grid.sumocfg')

    build_net(args.nx, args.ny, args.length, net_path)
    net = sumolib.net.readNet(net_path)
    n_flows = write_routes(net, rou_path)
    write_cfg(cfg_path, 'grid.net.xml', 'grid.rou.xml')
    tls = [t.getID() for t in net.getTrafficLights()]
    print(f'Built {args.nx}x{args.ny} grid: {len(tls)} signals {sorted(tls)}, {n_flows} flows')
    print(f'Config: {cfg_path}')


if __name__ == '__main__':
    main()
