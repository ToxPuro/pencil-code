# Running

* `cmake -DBUILD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=samples/les -DDSL_MODULE_DIR=../samples/les .. && make -j && ./les`

# Profiling (NVIDIA)

* `cmake -DBUILD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=samples/les -DDSL_MODULE_DIR=../samples/les .. && make -j && ncu  --profile-from-start off -f --set full -o report.profile ./les`