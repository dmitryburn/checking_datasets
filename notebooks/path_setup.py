import sys
import os

def setup_paths():
    parent_dir = os.path.abspath(os.path.join('..', 'src'))
    
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
