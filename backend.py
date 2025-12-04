import multiprocessing
import time
import os
import psutil
import traceback
from mps_sim import QPU
import numpy as np

# Worker function to run in a separate process
def simulation_worker(config, user_code, status_queue, stop_event):
    """
    config: dict with 'bond_dim', 'num_qubits', 'core_id'
    user_code: str, python code to execute
    status_queue: multiprocessing.Queue to send updates
    stop_event: multiprocessing.Event to signal stop
    """
    try:
        # Initialize QPU
        qpu = QPU(config['num_qubits'], config['bond_dim'])

        # Wrapper for QPU to check stop signal
        class QPUWrapper:
            def __init__(self, qpu_instance, stop_event):
                self._qpu = qpu_instance
                self._stop_event = stop_event

            def __getattr__(self, name):
                if self._stop_event.is_set():
                    raise InterruptedError("Simulation stopped by user")
                return getattr(self._qpu, name)

        # Let's add reporting to the wrapper
        class ReportingQPUWrapper(QPUWrapper):
            def __init__(self, qpu_instance, stop_event, queue, core_id):
                super().__init__(qpu_instance, stop_event)
                self._queue = queue
                self._core_id = core_id
                self._last_report = time.time()
                self._report_interval = 0.5 # seconds

            def __getattr__(self, name):
                if self._stop_event.is_set():
                    raise InterruptedError("Simulation stopped")

                # Check if it's time to report
                if time.time() - self._last_report > self._report_interval:
                    self._report()

                return getattr(self._qpu, name)

            def _report(self):
                # Send gate count
                try:
                    stats = {
                        'core_id': self._core_id,
                        'gate_count': self._qpu.gate_count,
                        'memory': self._qpu.memory_usage(),
                        'finished': False
                    }
                    # Non-blocking put
                    if not self._queue.full():
                         self._queue.put(stats)
                    self._last_report = time.time()
                except:
                    pass

        qpu_wrapper = ReportingQPUWrapper(qpu, stop_event, status_queue, config['core_id'])

        local_scope = {
            'qpu': qpu_wrapper,
            'np': np,
            'time': time
        }

        # Execute User Code
        exec(user_code, local_scope)

        # Final report
        status_queue.put({
            'core_id': config['core_id'],
            'gate_count': qpu.gate_count,
            'memory': qpu.memory_usage(),
            'finished': True
        })

    except InterruptedError:
        status_queue.put({'core_id': config['core_id'], 'error': 'Stopped', 'finished': True})
    except Exception as e:
        status_queue.put({'core_id': config['core_id'], 'error': str(e) + "\n" + traceback.format_exc(), 'finished': True})

class SimulationManager:
    def __init__(self):
        self.processes = []
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.start_time = None
        self.stats = {}

    def start_simulation(self, code, bond_dim, num_qubits, num_cores):
        if self.is_running():
            self.stop_simulation()

        self.stop_event.clear()
        self.processes = []
        self.queue = multiprocessing.Queue() # Fresh queue
        self.stats = {i: {'gate_count': 0, 'memory': 0, 'finished': False} for i in range(num_cores)}
        self.start_time = time.time()

        for i in range(num_cores):
            config = {
                'bond_dim': bond_dim,
                'num_qubits': num_qubits,
                'core_id': i
            }
            p = multiprocessing.Process(
                target=simulation_worker,
                args=(config, code, self.queue, self.stop_event)
            )
            p.daemon = True
            p.start()
            self.processes.append(p)

    def stop_simulation(self):
        self.stop_event.set()
        # Give them a moment to stop gracefully via wrapper check
        time.sleep(0.1)
        for p in self.processes:
            if p.is_alive():
                p.terminate()
        self.processes = []

    def is_running(self):
        return any(p.is_alive() for p in self.processes)

    def get_status(self):
        # Consume queue
        while not self.queue.empty():
            try:
                data = self.queue.get_nowait()
                cid = data['core_id']
                if 'error' in data:
                    self.stats[cid]['error'] = data['error']
                    self.stats[cid]['finished'] = True
                else:
                    self.stats[cid]['gate_count'] = data['gate_count']
                    self.stats[cid]['memory'] = data['memory']
                    self.stats[cid]['finished'] = data['finished']
            except:
                break

        # Aggregate
        total_gates = sum(s['gate_count'] for s in self.stats.values())
        total_mem = sum(s['memory'] for s in self.stats.values())
        active_cores = sum(1 for p in self.processes if p.is_alive())

        elapsed = 0
        if self.start_time:
            elapsed = time.time() - self.start_time

        gates_per_sec = total_gates / elapsed if elapsed > 0 else 0

        num_cores = len(self.stats)
        gates_per_sec_per_core = (gates_per_sec / num_cores) if num_cores > 0 else 0

        return {
            'elapsed': elapsed,
            'total_gates': total_gates,
            'gates_per_sec_total': gates_per_sec,
            'gates_per_sec_core': gates_per_sec_per_core,
            'total_memory': total_mem,
            'active_cores': active_cores,
            'errors': [s['error'] for s in self.stats.values() if 'error' in s]
        }
