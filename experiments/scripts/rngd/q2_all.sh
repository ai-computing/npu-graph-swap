# Q2: run q2_measure.py for each FXB in turn (fresh process each), then exit and release the card.
M=$(ls -d /mnt/store/hub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
mkdir -p /mnt/store/ngs/results
for lv in "$@"; do
  python3 /opt/ngs/q2_measure.py $M /mnt/store/ngs/fxb/l8_len1024_$lv.fxb /mnt/store/ngs/results/q2_$lv.json > /mnt/store/ngs/results/q2_$lv.log 2>&1
  sed 's/\x1b\[[0-9;]*m//g' /mnt/store/ngs/results/q2_$lv.log | grep -iE "RESULT|Error|Traceback|block|memory|GiB|MiB|bucket|kv" | grep -v "^RESULT" | head -40 | cut -c1-300
  grep "^RESULT" /mnt/store/ngs/results/q2_$lv.log | cut -c1-1200
done
