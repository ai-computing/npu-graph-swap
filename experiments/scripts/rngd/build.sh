# Build Llama-3.1-8B FXBs (max-model-len 1024) at optim levels O0..O3 concurrently; time each. No device needed.
M=$(ls -d /mnt/store/hub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
O=/mnt/store/ngs/fxb; mkdir -p $O; cd /tmp; : > $O/build_times.txt
python3 -c "import importlib.metadata as m; print({p: m.version(p) for p in ('furiosa-llm','furiosa-torch','torch')})" > $O/versions.txt
( while sleep 60; do echo "$(date +%T) load=$(cut -d' ' -f1 /proc/loadavg) rss_gb=$(ps -eo rss | awk '{s+=$1} END {print int(s/1048576)}')"; done ) > $O/build_monitor.txt &
MON=$!
pids=""
for lv in O0 O1 O2 O3; do
  ( s=$(date +%s); fxb build $M $O/l8_len1024_$lv.fxb --max-model-len 1024 -O $lv --build-report --concurrency 4 > $O/build_$lv.log 2>&1; rc=$?
    echo "$lv rc=$rc build_s=$(( $(date +%s) - s ))" >> $O/build_times.txt ) &
  pids="$pids $!"
done
wait $pids; kill $MON
ls -la $O; cat $O/build_times.txt
