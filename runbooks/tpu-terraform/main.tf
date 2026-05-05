terraform {
  required_version = ">= 1.5"

  # google-beta is required: the v2 TPU API (google_tpu_v2_vm) only exists in
  # the beta provider. The stable hashicorp/google provider still ships only
  # the deprecated google_tpu_node resource.
  required_providers {
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
  }
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Up to var.vm_count single-host TPU VMs (default: v6e-1, 1 chip, 32 GB HBM
# each). Each VM holds one model's weights at a time and rotates them; see
# ../TPU_RUN.md for the per-VM model assignments.
#
# Spot pricing: scheduling_config.spot is the TPU-specific knob (different
# from compute_instance.scheduling.preemptible — see Google Cloud's TPU Spot
# VMs docs).
resource "google_tpu_v2_vm" "belebele" {
  provider = google-beta
  count    = var.vm_count

  name             = "${var.vm_name_prefix}-${count.index}"
  zone             = var.zone
  runtime_version  = var.runtime_version
  accelerator_type = var.accelerator_type
  description      = "thesis-eval BELEBELE + heavy §9 generation worker"

  # Pin the spot/on-demand scheduling explicitly. Empty block yields
  # on-demand; spot=true enables preemptible spot pricing.
  scheduling_config {
    spot = var.use_spot
  }

  # Optional: tag with the labels GCP supports for billing/grouping.
  labels = {
    workload = "thesis-eval"
    purpose  = "belebele-heavy-gen"
  }
}
