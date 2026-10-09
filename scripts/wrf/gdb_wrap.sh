#!/bin/bash
# run the MPI ranks listed in GDB_RANKS (comma list) of wrf.exe under gdb (backtrace at the crash)
r=${PMI_RANK:-${MPI_LOCALRANKID:--1}}
if [[ ",${GDB_RANKS:-22}," == *",$r,"* ]]; then
  exec gdb -batch -ex "set pagination off" -ex run -ex "bt 30" --args ./wrf.exe > gdb_rank_$r.txt 2>&1
else
  exec ./wrf.exe
fi
