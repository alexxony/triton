import subprocess, textwrap
sh = r'''
set -u
S=/content/build_status.json
st(){ python3 - "$@" <<PY
import json,sys,time
json.dump({"stage":sys.argv[1],"t":int(time.time()),"detail":sys.argv[2] if len(sys.argv)>2 else ""},open("/content/build_status.json","w"))
PY
}
st start
cd /content && rm -rf triton && mkdir triton && cd triton
git init -q && git remote add origin https://github.com/triton-lang/triton.git
git fetch -q --depth 1 origin 93ba8872b6d783665af1cf662ebd48bda49e60b9 || { st fail "fetch"; exit 1; }
git -c advice.detachedHead=false checkout -q FETCH_HEAD
git rev-parse HEAD > /content/build_sha.txt
st deps
pip install -q cmake ninja pybind11 setuptools wheel lit 2>&1 | tail -3
st build "$(date +%s)"
export MAX_JOBS=8
mkdir -p /content/wheelhouse
pip wheel . --no-deps -w /content/wheelhouse -v > /content/build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then st fail "pip rc=$rc"; exit $rc; fi
ls -la /content/wheelhouse > /content/wheel_ls.txt
st done "$(ls /content/wheelhouse)"
'''
open('/content/build.sh','w').write(sh)
subprocess.Popen(['nohup','bash','/content/build.sh'],stdout=open('/content/build_outer.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print("launched")
