#!/usr/bin/env bash
# Retries `terraform apply -auto-approve` while the failures are TPU capacity
# errors (Error code 8, "no more capacity in the zone"). Bails out on any
# other failure so we don't loop forever on real bugs.
set -uo pipefail

cd "$(dirname "$0")"

attempt=1
max_attempts=30
sleep_seconds=120

while true; do
  ts=$(date '+%Y-%m-%dT%H:%M:%S%z')
  echo "=== attempt ${attempt} @ ${ts} ==="
  out=$(terraform apply -auto-approve -no-color 2>&1)
  rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "=== terraform apply SUCCESS at attempt ${attempt} ==="
    break
  fi

  if echo "$out" | grep -q "no more capacity in the zone"; then
    capacity_count=$(echo "$out" | grep -c "no more capacity in the zone")
    echo "=== attempt ${attempt}: ${capacity_count} VM(s) still capacity-blocked ==="
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "=== giving up after ${max_attempts} attempts (~$((max_attempts * sleep_seconds / 60)) min) ==="
      exit 2
    fi
    attempt=$((attempt + 1))
    sleep "$sleep_seconds"
    continue
  fi

  echo "=== non-capacity failure on attempt ${attempt}; aborting retry ==="
  echo "$out" | tail -40
  exit 1
done

echo
zone=$(grep -E '^zone' terraform.tfvars 2>/dev/null | sed -E 's/.*= *"([^"]+)".*/\1/')
zone=${zone:-europe-west4-a}
echo "=== final state in ${zone} ==="
gcloud compute tpus tpu-vm list --project=mestrado-495311 --zone="$zone"
echo
echo "=== terraform outputs ==="
terraform output
