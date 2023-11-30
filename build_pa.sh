echo "Compile perf_assist..."
aarch64-linux-android30-clang++ perf_assist.cpp -std=c++14 -o out/perf_assist -static-libstdc++
echo "Done."
