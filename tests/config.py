# adding python path to be able to have access to mcrpy package from sibling folder tests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))