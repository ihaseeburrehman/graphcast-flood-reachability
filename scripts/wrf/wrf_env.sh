# Runtime environment for the author's own WRF 4.5.2 / WPS 4.5 build on Atos (libraries in ~/WRF/Libs)
L=$HOME/WRF/Libs
export LD_LIBRARY_PATH=$(ls -d $L/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}
export PATH=$L/MPICH/bin:$L/NETCDF/bin:$PATH
export WPS=$HOME/WRF/WPS-4.5 WRFRUN=$HOME/WRF/WRFV4.5.2/run GEOG=/ec/res4/hpcperm/lux0804/WPS_GEOG
export W=/ec/res4/scratch/lux0804/gc_flood_predictability/wrf
export CODE=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code
# MPICH 4.2.1 (~/WRF/Libs/MPICH) was linked against libslurm.so.41; Atos now has .43. Single-node runs use
# hydra's fork launcher (no Slurm calls), so only the library has to load: compat symlink .41 -> system .43.
export LD_LIBRARY_PATH=/ec/res4/hpcperm/lux0804/wrf_compat:$LD_LIBRARY_PATH
export HYDRA_LAUNCHER=fork
# WRF allocates large automatic arrays on the stack: without this, 15 of 128 tasks segfaulted at step 1 on the
# 420x340 domain (the v_cfl "blow-up" messages were a side effect)
ulimit -s unlimited
export OMP_STACKSIZE=1G
