# Pilot run notes and instructions

# Important files:

* `acc-runtime/samples/mhd_modular/mhdsolver.ac`
> **NOTE**: `dsx`, `dsy`, and `dsz` are hardcoded here. If grid dimensions are changed in config file (f.ex. `config/astaroth.conf`), then they must be manually changed here also, otherwise grid spacing in integration kernel will be incorrect.

* `config/samples/subsonic_forced_nonhelical_turbulence/astaroth.conf`
> One of the configs, should be compatible with the mahti 4k snapshot

* `config/astaroth.conf`
> The default config if no `--config <path-to-config>` is passed to `./ac_run_mpi`

# Issues

* Weird issues when writing slices or mesh (device context invalid, etc)
> Try writing without MPI IO by setting `#define USE_POSIX_IO (1)` in `astaroth/src/core/grid.cc` in `acGridWriteMeshToDiskLaunch` or `acGridWriteSlicesToDiskLaunch`, or both.
> You can also try without async IO by commenting out `threads.push_back(std::move(std::thread(write_async, host_buffer, count))); // Async, threaded` at the end of these functions and uncommenting `write_async(host_buffer, count, device->id); // Synchronous, non-threaded` or similar. However, note that this results in worse performance due to IO being blocking.

* Distributed mesh writing and collective slice writing?
> Replace `acGridWriteSlicesToDiskLaunch` calls in `standalone_mpi/main.cc` with `acGridWriteSlicesToDiskCollectiveSynchronous`.

# Manual workflow without scripts (see `scripts/gen_rundir.sh` for automating parts of this)

Variables:
    * `CONFIG` - path to a `.config` file, for example `config/samples/subsonic_forced_nonhelical_turbulence/astaroth.conf`.
    * `ASTAROTH` - path to astaroth directory

1) Modify `CONFIG`

2) Modify `acc-runtime/samples/mhd_modular/mhdsolver.ac` s.t. dsx are equivalent with those defined in `CONFIG`

3) Build `cd astaroth && mkdir build && cd build && cmake .. && make -j`

4) Create a run directory (preferably in scratch or flash): `cd $SCRATCH && mkdir some-run-name-here && cd some-run-name-here`

5) Copy the necessary files. For example the 4k run `cp $ASTAROTH/build/ac_run_mpi .` (executable), `cp $ASTAROTH/config/samples/subsonic_forced_nonhelical_turbulence/astaroth.conf .` (config file), and `cp $ASTAROTH/pilot/pilot-4096.sh` (batch script).

6) Modify the three files as needed. Ensure that the batch script has correct time allocation, number of devices (f.ex. 4096 case: 8 devices per nodes, one task per device, ntasks/ndevices_per_node nodes, so `--gres=gpu:8`, `--ntasks=4096`, and `--nodes=512`), and that the config and varfile directories are correct.  For example:
> To restart from varfile: `srun ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/flash/project_462000120/striped_dir/var.dat` (IMPORTANT: must have `AC_start_step = 0` in `CONFIG`)
> To restart from a snapshot: `srun ./ac_run_mpi --config ./astaroth.conf --from-snapshot` (IMPORTANT: must have `AC_start_step = -1` in `CONFIG`)
> To start a randomized run: `srun ./ac_run_mpi --config ./astaroth.conf --run-init-kernel` (IMPORTANT: must have `AC_start_step = 0` in `CONFIG`)

7) Queue the run, f.ex. `sbatch pilot-4096.sh`

# Helper scripts

* You can use `pilot/plot-distributed.py` and `pilot/plot-collective.py` snippets to quickly visualize slices. There are some more advanced visualization scripts in `astaroth/analysis`.