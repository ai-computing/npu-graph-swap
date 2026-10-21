# Emit a Job manifest for npu-graph-swap work. usage: mkjob.py NAME RNGD(0|1) DEADLINE_S CMD
# NODE (env) is the RNGD node's Kubernetes hostname; the store PVC and image follow the lab's existing jobs.
import os, sys, json
name, rngd, deadline, cmd = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), sys.argv[4]
wrap = ("mkdir -p /mnt/store/ngs/logs && echo \"start $(date +%s) $(hostname)\" > /mnt/store/ngs/logs/{n}.meta && "
        "(command -v aarch64-linux-gnu-gcc >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y -qq --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross libc6-dev >/dev/null)) && "
        "mkdir -p ~/.cache/furiosa/compiler && set -o pipefail; ( {c} ) 2>&1 | tee /mnt/store/ngs/logs/{n}.out; rc=$?; "
        "echo \"end $(date +%s) rc=$rc\" >> /mnt/store/ngs/logs/{n}.meta; exit $rc").format(n=name, c=cmd)
res = {"requests": {"cpu": "16", "memory": "96Gi"}}
c = {"name": "run", "image": "furiosaai/furiosa-llm:latest", "imagePullPolicy": "IfNotPresent",
     "command": ["bash", "-c", wrap],
     "env": [{"name": k, "value": v} for k, v in (("PYTHONUNBUFFERED", "1"), ("HF_HUB_OFFLINE", "1"),
             ("HF_HOME", "/mnt/store/hub"), ("PYTHONPATH", "/opt/ngs"))],
     "resources": res,
     "volumeMounts": [{"name": "store", "mountPath": "/mnt/store"}, {"name": "shm", "mountPath": "/dev/shm"},
                      {"name": "src", "mountPath": "/opt/ngs", "readOnly": True}]}
spec = {"restartPolicy": "Never", "containers": [c],
        "volumes": [{"name": "store", "persistentVolumeClaim": {"claimName": "model-store"}},
                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                    {"name": "src", "configMap": {"name": "ngs-src"}}]}
if rngd:
    res["limits"] = {"furiosa.ai/rngd": "1"}
    c["securityContext"] = {"privileged": True}
    spec["nodeSelector"] = {"kubernetes.io/hostname": os.environ.get("NODE", "npu-node-1")}
    # privileged containers see every RNGD on the node; a non-privileged init container records the card the
    # device plugin assigned, and the measurement uses only that card.
    spec["initContainers"] = [{"name": "assigned-rngd", "image": "busybox:1.36", "imagePullPolicy": "IfNotPresent",
        "command": ["sh", "-c", "ls /dev/rngd | grep -oE '^npu[0-9]+' | sort -u | head -1 > /run/ngs/rngd && echo assigned $(cat /run/ngs/rngd)"],
        "resources": {"limits": {"furiosa.ai/rngd": "1"}}, "volumeMounts": [{"name": "rngd", "mountPath": "/run/ngs"}]}]
    spec["volumes"].append({"name": "rngd", "emptyDir": {}})
    c["volumeMounts"].append({"name": "rngd", "mountPath": "/run/ngs"})
job = {"apiVersion": "batch/v1", "kind": "Job",
       "metadata": {"name": name, "namespace": "llm-serving", "labels": {"ngs-exp": "true"}},
       "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 86400, "activeDeadlineSeconds": deadline,
                "template": {"metadata": {"labels": {"ngs-exp": "true", "app": name}}, "spec": spec}}}
print(json.dumps(job))
