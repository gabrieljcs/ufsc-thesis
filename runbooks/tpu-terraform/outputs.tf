output "vm_names" {
  description = "TPU VM names provisioned by this stack."
  value       = google_tpu_v2_vm.belebele[*].name
}

output "ssh_commands" {
  description = "Copy-paste SSH commands, one per VM."
  value = [
    for vm in google_tpu_v2_vm.belebele :
    "gcloud compute tpus tpu-vm ssh ${vm.name} --project=${var.project_id} --zone=${var.zone}"
  ]
}

output "scp_pull_outputs_spec_commands" {
  description = "Copy-paste scp commands to pull outputs/spec/ back from each VM."
  value = [
    for vm in google_tpu_v2_vm.belebele :
    "gcloud compute tpus tpu-vm scp --recurse --project=${var.project_id} --zone=${var.zone} ${vm.name}:~/ufsc-thesis/eval/outputs/spec ./eval/outputs/"
  ]
}

output "scp_pull_outputs_generations_commands" {
  description = "Copy-paste scp commands to pull outputs/generations/ back from each VM."
  value = [
    for vm in google_tpu_v2_vm.belebele :
    "gcloud compute tpus tpu-vm scp --recurse --project=${var.project_id} --zone=${var.zone} ${vm.name}:~/ufsc-thesis/eval/outputs/generations ./eval/outputs/"
  ]
}

output "summary" {
  description = "One-line summary printed at apply time."
  value = format(
    "Provisioned %d × %s in %s (%s). vm_count=%d use_spot=%s",
    var.vm_count,
    var.accelerator_type,
    var.zone,
    var.runtime_version,
    var.vm_count,
    var.use_spot,
  )
}
