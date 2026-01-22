#!/bin/bash -l
rm -r /home/hey4/RICH-fwrk/build
module purge && module restore new_rich_build			
/home/hey4/RICH-fwrk/config.py --problem=ShockTube --mpi
make -C /home/hey4/RICH-fwrk
