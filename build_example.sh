echo Compiling $1_aarch64...
aarch64-linux-android32-clang++ examples/$1.cpp utils/Utils.cpp utils/GraphUtils.cpp utils/CommonGraphOptions.cpp -I. -Iinclude -std=c++14 -Wl,--whole-archive -Lbuild/  -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -L. -o ../../out/$1_aarch64 -static-libstdc++ -pie -DARM_COMPUTE_CL
echo Done.


