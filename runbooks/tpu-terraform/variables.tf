variable "project_id" {
  type        = string
  description = "GCP project hosting the TRC TPU quotas."
}

variable "region" {
  type        = string
  description = "GCP region containing var.zone. Inferred from zone if not overridden."
  default     = "europe-west4"
}

variable "zone" {
  type        = string
  description = <<-EOT
    TPU zone. Suggested:
      - europe-west4-a → v6e (Trillium) spot
      - us-east1-d     → v6e spot, US-side latency
      - europe-west4-b → v5e spot
      - us-central2-b  → v4 on-demand or spot (set use_spot=false for on-demand)
  EOT
  default     = "europe-west4-a"
}

variable "accelerator_type" {
  type        = string
  description = <<-EOT
    Accelerator slice. Recommended single-host shapes:
      - v6e-1  (1 chip,  32 GB HBM)  ← default
      - v5litepod-4  (4 chips, 64 GB HBM total) — use only if v6e unavailable
      - v4-8   (4 chips, 128 GB HBM total) — for us-central2-b on-demand
    Note: predict-belebele/generate-targets use single-device transformers
    on XLA, so multi-chip slices add no speed without code changes.
  EOT
  default     = "v6e-1"
}

variable "runtime_version" {
  type        = string
  description = <<-EOT
    TPU runtime image. Match this to the accelerator family:
      - v6e          → v2-alpha-tpuv6e
      - v5e/v5litepod → v2-alpha-tpuv5-lite
      - v4           → tpu-ubuntu2204-base
    Check current options with: gcloud compute tpus versions list --zone=<ZONE>
  EOT
  default     = "v2-alpha-tpuv6e"
}

variable "vm_count" {
  type        = number
  description = "Number of parallel TPU VMs to provision. 1-4 recommended; one heavy §9 model per VM."
  default     = 4

  validation {
    condition     = var.vm_count >= 1 && var.vm_count <= 8
    error_message = "vm_count must be between 1 and 8 (TRC quotas / debugging sanity)."
  }
}

variable "use_spot" {
  type        = bool
  description = "Spot pricing (preemptible). True for spot v5e/v6e quota; false for on-demand v4 in us-central2-b."
  default     = true
}

variable "vm_name_prefix" {
  type        = string
  description = "Name prefix for the TPU VMs; final names are <prefix>-<index>, e.g. thesis-belebele-0."
  default     = "thesis-belebele"
}
