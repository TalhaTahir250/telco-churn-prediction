# src/app/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.app.main:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = True,
    )