curl --silent --show-error --fail-with-body --max-time 120 -D t4-evidence/baseline/ready.headers -o t4-evidence/baseline/ready.body.json -w %\{http_code\}\\n http://127.0.0.1:8080/ready 
