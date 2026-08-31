import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "g1_piano_teaching" / "src"))
from g1_piano.recorder.state_recorder import main

if __name__ == "__main__":
    main()
