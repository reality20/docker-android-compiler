import gradio as gr
import time
import multiprocessing
from backend import SimulationManager

# Global Manager
manager = SimulationManager()

def start_run(bond_dim, num_qubits, num_cores, code):
    if not code.strip():
        return "No code provided", "0", "0", "0", "0", ""

    try:
        manager.start_simulation(code, int(bond_dim), int(num_qubits), int(num_cores))
        return "Simulation Started", "0", "0", "0", "0", ""
    except Exception as e:
        return f"Error: {e}", "0", "0", "0", "0", ""

def stop_run():
    manager.stop_simulation()
    return "Simulation Stopped"

def update_status():
    status = manager.get_status()

    state_str = "Running" if manager.is_running() else "Idle"
    if status['errors']:
        state_str = f"Error: {status['errors'][0]}"

    elapsed = f"{status['elapsed']:.2f} s"
    gps_core = f"{status['gates_per_sec_core']:.2e}"
    mem = f"{status['total_memory'] / 1024**2:.2f} MB"
    total_gates = f"{status['total_gates']}"
    output_log = status['output']

    return state_str, elapsed, gps_core, mem, total_gates, output_log

default_code = """# Example Simulation: 3D Quantum Walk
# This demo showcases the 3D Hypercube topology.
import random
import time

core = qpu._core_id
print(f"Core {core}: Starting 3D Quantum Walk on {qpu.num_qubits} qubits.")
print(f"Topology: {qpu.L} x {qpu.W} x {qpu.H}")

# Check neighbors of qubit 0
print(f"Core {core}: Neighbors of 0: {[i for i in range(qpu.num_qubits) if qpu._are_neighbors(0, i)]}")

# 1. Initialize
# Apply Hadamard to all to create superposition
for i in range(qpu.num_qubits):
    qpu.h(i)

# 2. Random Walk in 3D
# We pick a qubit and try to interact with a neighbor
steps = 1000
for s in range(steps):
    q = random.randint(0, qpu.num_qubits - 1)

    # Try to find a valid neighbor
    neighbors = []
    # Since we don't have a public neighbor list, we scan (slow for demo, but checks logic)
    # Actually, let's just pick another random one and try-catch

    target = random.randint(0, qpu.num_qubits - 1)

    try:
        if q != target:
            qpu.cx(q, target)
    except ValueError:
        # Not neighbors
        pass

# 3. Measurement
m = qpu.measure(0)
print(f"Core {core}: Measurement of q0 = {m}")

print(f"Core {core}: Finished.")
"""

with gr.Blocks(title="QPU Simulator") as demo:
    gr.Markdown("# QPU Simulator (MPS Backend - 3D Hypercube)")
    gr.Markdown("This simulator uses a Matrix Product State backend optimized for a 3D Hypercube topology using Numba.")

    with gr.Row():
        with gr.Column(scale=1):
            bond_dim = gr.Number(label="Bond Dimension (Fixed to 2)", value=2, precision=0, interactive=False)
            num_qubits = gr.Number(label="Number of Qubits", value=27, precision=0) # 3x3x3
            # Default to all cores
            default_cores = multiprocessing.cpu_count()
            num_cores = gr.Number(label="Number of Cores", value=default_cores, precision=0)

            run_btn = gr.Button("Run Simulation", variant="primary")
            kill_btn = gr.Button("Kill Simulation", variant="stop")

        with gr.Column(scale=2):
            code_input = gr.Code(label="Simulation Code (Python)", value=default_code, language="python", lines=25)

    with gr.Row():
        status_display = gr.Textbox(label="Status", value="Idle", interactive=False)
        time_display = gr.Textbox(label="Elapsed Time", value="0 s", interactive=False)
        gps_display = gr.Textbox(label="Gates/s (Single Core)", value="0", interactive=False)
        mem_display = gr.Textbox(label="Est. Memory", value="0 MB", interactive=False)
        gates_display = gr.Textbox(label="Total Gates Executed", value="0", interactive=False)

    with gr.Row():
        output_display = gr.Code(label="Output Log", value="", language=None, lines=10, interactive=False)

    # Event handlers
    run_btn.click(
        fn=start_run,
        inputs=[bond_dim, num_qubits, num_cores, code_input],
        outputs=[status_display, time_display, gps_display, mem_display, gates_display, output_display]
    )

    kill_btn.click(
        fn=stop_run,
        inputs=[],
        outputs=[status_display]
    )

    # Timer for updates
    timer = gr.Timer(0.5)
    timer.tick(update_status, inputs=[], outputs=[status_display, time_display, gps_display, mem_display, gates_display, output_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
