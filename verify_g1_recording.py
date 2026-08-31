import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "g1_piano_teaching" / "src"))
from g1_piano.analysis.verify_recording import main

if __name__ == "__main__":
    raise SystemExit(main())
