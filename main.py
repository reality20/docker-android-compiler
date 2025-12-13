from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import io
import contextlib
import traceback
from quantum_engine import QPU

app = FastAPI()

class RunRequest(BaseModel):
    code: str
    num_cores: int

class RunResponse(BaseModel):
    output: str
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/run", response_model=RunResponse)
def run_code(request: RunRequest):
    # Capture stdout
    stdout_buffer = io.StringIO()

    error_msg = None

    try:

        qpu = QPU(request.num_cores)


        exec_globals = {
            "qpu": qpu,
            "QPU": QPU,
            "range": range,
            "len": len,
            "print": print
        }

        with contextlib.redirect_stdout(stdout_buffer):

            exec(request.code, exec_globals)
    except Exception:
        traceback.print_exc()
        error_msg = traceback.format_exc()

    return RunResponse(output=stdout_buffer.getvalue(), error=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
