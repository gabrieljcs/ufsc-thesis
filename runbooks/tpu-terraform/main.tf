terraform {
  required_version = ">= 1.5"

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

# --- Outbound internet for IP-less TPU VMs --------------------------------
#
# The google-beta provider has a known bug serializing both `network_config`
# and `network_configs` on `google_tpu_v2_vm`, which the Cloud TPU API
# rejects ("NetworkConfig and NetworkConfigs fields are exclusive"). Easiest
# workaround: omit the block entirely. VMs come up with internal IPs only.
#
# Cloud NAT gives those internal-only VMs outbound internet (HF, PyPI,
# GitHub, the libtpu wheel index, etc.) without external IPs. SSH from the
# laptop still works through `gcloud ssh`'s IAP tunnel.

resource "google_compute_router" "tpu" {
  provider = google-beta
  name     = "thesis-tpu-router-${var.region}"
  network  = "default"
  region   = var.region
}

resource "google_compute_router_nat" "tpu" {
  provider                           = google-beta
  name                               = "thesis-tpu-nat-${var.region}"
  router                             = google_compute_router.tpu.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = false
    filter = "ERRORS_ONLY"
  }
}

# Firewall rule allowing inbound SSH from IAP's source range. Required so
# that `gcloud compute tpus tpu-vm ssh` can tunnel to a VM without an
# external IP. The IAP source range (35.235.240.0/20) is fixed and
# documented at https://cloud.google.com/iap/docs/using-tcp-forwarding.
resource "google_compute_firewall" "iap_ssh" {
  provider = google-beta
  name     = "thesis-tpu-iap-ssh"
  network  = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
  direction     = "INGRESS"
}

# --- TPU VMs --------------------------------------------------------------

resource "google_tpu_v2_vm" "belebele" {
  provider = google-beta
  count    = var.vm_count

  name             = "${var.vm_name_prefix}-${count.index}"
  zone             = var.zone
  runtime_version  = var.runtime_version
  accelerator_type = var.accelerator_type
  description      = "thesis-eval BELEBELE + heavy §9 generation worker"

  scheduling_config {
    spot = var.use_spot
  }

  labels = {
    workload = "thesis-eval"
    purpose  = "belebele-heavy-gen"
  }

  # Make sure NAT is up before we boot VMs that depend on outbound internet.
  depends_on = [
    google_compute_router_nat.tpu,
    google_compute_firewall.iap_ssh,
  ]
}
