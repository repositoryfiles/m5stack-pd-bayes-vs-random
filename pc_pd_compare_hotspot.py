"""Compare Bayesian optimization and random search on an inverted pendulum.
Install: py -m pip install numpy scipy scikit-learn
Run: py pc_pd_compare_hotspot.py
Protocol: GET /gains, POST /result, GET /status. Global trial IDs 1..180.
One timestamped CSV per run. Stop with Ctrl+C. No automatic restart/resume.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import threading
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

HOST, PORT = '0.0.0.0', 8000
TRIALS_PER_SET = 30
INITIAL_TRIALS = 6
METHODS = ['bayes', 'random', 'random', 'bayes', 'bayes', 'random']
KP_BOUNDS = (0.0, 80.0)
KD_BOUNDS = (0.0, 4.0)
SEED = 20261111
MAX_TRIALS = TRIALS_PER_SET * len(METHODS)
CSV_PATH = Path(__file__).with_name('pd_compare_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
FIELDS = ['datetime', 'method', 'set', 'trial_in_set', 'trial', 'kp', 'kd', 'reason', 'score',
          'elapsed_ms', 'samples', 'initial_theta', 'final_theta', 'min_theta',
          'max_theta', 'min_abs_theta', 'rms', 'eval_samples', 'upright_ms',
          'rise_ms', 'max_command', 'max_dt_ms', 'released']

# Use the same fixed six initial gains as the original Bayesian script for every Bayesian set.
init_rng = np.random.default_rng(SEED)
initial = []
for i in range(INITIAL_TRIALS):
    kp = KP_BOUNDS[0] + (KP_BOUNDS[1] - KP_BOUNDS[0]) * (i + init_rng.random()) / INITIAL_TRIALS
    kd = KD_BOUNDS[0] + (KD_BOUNDS[1] - KD_BOUNDS[0]) * (i + init_rng.random()) / INITIAL_TRIALS
    initial.append((float(kp), float(kd)))
init_rng.shuffle(initial)

# Independent random streams for each set, so set switching doesn't affect other sets.
set_rngs = [np.random.default_rng(SEED + 1000 + i) for i in range(len(METHODS))]

with CSV_PATH.open('x', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerow(FIELDS)

lock = threading.Lock()
results = []
pending = None


def objective(d):
    """Exactly the same score as the original program."""
    reason = str(d['reason'])
    elapsed = max(0.0, min(3.0, float(d['elapsed_ms']) / 1000.0))
    if reason == 'Time limit' and bool(d['released']) and float(d['eval_samples']) >= 190:
        return max(0.0, min(9.0, float(d['rms'])))
    if reason == 'Rise failure':
        return 25.0 + min(10.0, max(0.0, float(d['min_abs_theta'])))
    return 15.0 + (3.0 - elapsed) * 2.0


def current_position():
    global_index = len(results)
    set_index, local_index = divmod(global_index, TRIALS_PER_SET)
    return set_index, local_index


def propose():
    set_index, local_index = current_position()
    method = METHODS[set_index]
    rng = set_rngs[set_index]
    if method == 'random':
        return (float(rng.uniform(*KP_BOUNDS)), float(rng.uniform(*KD_BOUNDS)))
    if local_index < INITIAL_TRIALS:
        return initial[local_index]

    # Only data from the current Bayesian set are used for its GP model.
    start = set_index * TRIALS_PER_SET
    subset = results[start:]
    x = np.array([[r['kp'] / 80.0, r['kd'] / 4.0] for r in subset], dtype=float)
    y = np.array([r['score'] for r in subset], dtype=float)
    kernel = (ConstantKernel(1.0, (1e-2, 1e3)) *
              Matern(length_scale=[0.3, 0.3], length_scale_bounds=(0.03, 3.0), nu=2.5)
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 100.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=SEED)
    try:
        gp.fit(x, y)
        candidates = rng.uniform(0.0, 1.0, size=(3000, 2))
        mean, std = gp.predict(candidates, return_std=True)
        std = np.maximum(std, 1e-9)
        improvement = np.min(y) - mean - 0.05
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        distances = np.sqrt(((candidates[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
        ei[np.min(distances, axis=1) < 0.025] = -1.0
        point = candidates[int(np.argmax(ei))]
    except Exception as e:
        print('GP fallback:', repr(e), flush=True)
        point = rng.uniform(0.0, 1.0, size=2)
    return float(point[0] * 80.0), float(point[1] * 4.0)


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, code=200):
        b = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        global pending
        with lock:
            if self.path == '/gains':
                if len(results) >= MAX_TRIALS:
                    self.reply({'status': 'finished', 'trials': len(results)})
                    return
                set_index, local_index = current_position()
                if pending is None:
                    pending = propose()
                    print('Set', set_index + 1, METHODS[set_index], 'trial', local_index + 1,
                          '(global', len(results) + 1, ')', 'kp', round(pending[0], 3),
                          'kd', round(pending[1], 3), flush=True)
                self.reply({'status': 'ready', 'trial': len(results) + 1,
                            'kp': pending[0], 'kd': pending[1]})
            elif self.path == '/status':
                if len(results) >= MAX_TRIALS:
                    self.reply({'status': 'finished', 'completed': len(results),
                                'max_trials': MAX_TRIALS})
                else:
                    si, li = current_position()
                    self.reply({'completed': len(results), 'max_trials': MAX_TRIALS,
                                'set': si + 1, 'trial_in_set': li + 1,
                                'method': METHODS[si]})
            else:
                self.send_error(404)

    def do_POST(self):
        global pending
        if self.path != '/result':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                self.reply({'error': 'invalid body size'}, 400)
                return
            d = json.loads(self.rfile.read(size))
            with lock:
                trial = int(d['trial'])
                if trial == len(results) and trial > 0:
                    self.reply({'status': 'duplicate', 'completed': len(results)})
                    return
                if trial != len(results) + 1 or trial > MAX_TRIALS or pending is None:
                    self.reply({'error': 'trial mismatch', 'expected': len(results) + 1}, 409)
                    return
                kp, kd = float(d['kp']), float(d['kd'])
                if abs(kp - pending[0]) > 0.0001 or abs(kd - pending[1]) > 0.0001:
                    self.reply({'error': 'gain mismatch'}, 409)
                    return
                for key in ('elapsed_ms', 'rms', 'min_abs_theta', 'eval_samples'):
                    if not math.isfinite(float(d[key])):
                        self.reply({'error': 'invalid ' + key}, 400)
                        return
                set_index, local_index = current_position()
                d['score'] = objective(d)
                row = {k: d.get(k, '') for k in FIELDS}
                row.update(datetime=datetime.now().isoformat(timespec='seconds'),
                           method=METHODS[set_index], set=set_index + 1,
                           trial_in_set=local_index + 1, trial=trial)
                with CSV_PATH.open('a', newline='', encoding='utf-8-sig') as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                results.append({'kp': kp, 'kd': kd, 'score': d['score']})
                pending = None
                print('Saved', trial, 'set', set_index + 1, METHODS[set_index],
                      'reason', d['reason'], 'score', round(d['score'], 4), flush=True)
                if local_index + 1 == TRIALS_PER_SET:
                    print('--- Set', set_index + 1, 'finished ---', flush=True)
                if len(results) == MAX_TRIALS:
                    print('ALL 180 TRIALS FINISHED. CSV:', CSV_PATH, flush=True)
                self.reply({'status': 'ok', 'completed': len(results)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            self.reply({'error': str(e)}, 400)


if __name__ == '__main__':
    print('Comparison:', METHODS, '30 trials per set;', MAX_TRIALS, 'total', flush=True)
    print('CSV:', CSV_PATH, flush=True)
    print('Bayes: 6 fixed initial + 24 EI; random: 30 independent uniform draws', flush=True)
    print('Trial IDs are global 1..180. Confirm the M5Stack code accepts IDs > 30.', flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()
