./build_example.sh $1
./build_pa.sh

echo "Push program on the mobile device..."
adb push out/perf_assist /data/local/tmp/
adb shell chmod +x /data/local/tmp/perf_assist

adb push out/$1_aarch64 /data/local/tmp/
adb shell chmod +x /data/local/tmp/$1_aarch64
echo "Done."
