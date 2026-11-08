"""Run the existing resident exact-geometry oracle under an external allocation.

The queue binds canonical parent coordinates and the pinned teacher context.
It stores evaluated forces in their actual orientation and never aliases forces
from a symmetry-equivalent geometry. Scheduling and termination remain external.
"""
from cathode_cvae.pool_scoring import main

if __name__ == '__main__':
    main()
