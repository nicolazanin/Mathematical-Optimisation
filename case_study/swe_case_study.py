import sys
import numpy as np
from gurobipy import GRB
import logging
import time

from utils.case_study_utils import get_airports

airports = get_airports('swe_airports.csv')
print(airports[0])