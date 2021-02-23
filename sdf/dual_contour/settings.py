"""Stores some global settings, mostly to save on a lot of repetitive arguments.
These are mostly of interest for demonstrating different variants of the algorithms as discussed
in the accompanying article."""

# In dual contouring, if true, crudely force the selected vertex to belong in the cell
CLIP = False
# In dual contouring, if true, apply boundaries to the minimization process finding the vertex for each cell
BOUNDARY = True
# In dual contouring, if true, apply extra penalties to encourage the vertex to stay within the cell
BIAS = True
# Strength of the above bias, relative to 1.0 strength for the input gradients
BIAS_STRENGTH = 0.01
