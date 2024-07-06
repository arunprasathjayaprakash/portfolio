provider "google" {
  credentials = "portfolio_project_key.json"
  project = var.project_id
  region = var.region
}

#resource "google_storage_bucket" "state" {
#  location = "us-east1"
#  name     = var.state_bucket
#  project = var.project_id
#  storage_class = "standard"
#  force_destroy = "true"
#  versioning {
#    enabled = false
#  }
#}

resource "google_container_cluster" "primary" {
  location = "us-east1"
  name     = "churn-prediction-cluster"
   deletion_protection = "false"

  node_pool {
    autoscaling {
      min_node_count = 1
      max_node_count = 1
    }

    node_config {
    machine_type="e2-small"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    disk_size_gb = 10
  }

  node_count = 1
}
}

resource "google_artifact_registry_repository" "default" {
  location = "us-east1"
  project = var.project_id
  format = "DOCKER"
  repository_id = "default"
}

resource "google_artifact_registry_repository_iam_binding" "admin" {
  project = google_artifact_registry_repository.default.project
  location = google_artifact_registry_repository.default.location
  repository = google_artifact_registry_repository.default.repository_id
  role = "roles/artifactregistry.writer"
  members = [
    "user:arunprasathjayaprakash12@gmail.com"
  ]
}

#resource "google_container_node_pool" "primary_nodes" {
#  cluster = google_container_cluster.primary.name
#  location = "us-east1"
#  node_count = 1
#
#  node_config {
#    machine_type = "e2-small"
#    oauth_scopes = [
#      "https://www.googleapis.com/auth/cloud-platform"
#    ]
#    disk_size_gb = 10
#
#  }
#  max_pods_per_node = 10
#}

#resource "google_service_account" "cloud_run_service_account" {
#  account_id   = "cloud-run-sa"
#  display_name = "Cloud Run Service Account"
#}


resource "google_cloud_run_service" "churn-docker" {
  location = "us-east1"
  name     = "churn-docker"

  template {
    spec {
      containers {
        image = "us-east1-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.default
.repository_id}/churn_image:v0.2"
        resources {
          limits = {
            memory = "256Mi"
            cpu    = "1"
          }
        }
        args = ["streamlit", "run", "scripts/app.py", "--server.port=8080"]
      }

      service_account_name = "portfolio-projects@complete-will-428105-h5.iam.gserviceaccount.com"
    }
  }
  traffic {
    percent = 100
  }
}

terraform {
  backend "gcs" {
    bucket = "portfolio_buckets_2024"
    credentials = "portfolio_project_key.json"
    prefix = "terraform/state_v1"
  }
}

output "project_id" {
  description = "gcp region"
  value       = var.project_id
}

output "state_bucket" {
  value = var.state_bucket
}

output "cluster_name" {
  value = var.cluster_name
}

output "k8s_version" {
  value = var.k8s_version
}

output "region" {
  value = var.region
}

output "artifact_registry_url" {
  value = "${google_artifact_registry_repository.default.location}-docker.pkg.dev/${var.project_id}/${
google_artifact_registry_repository.default.repository_id}"
}

