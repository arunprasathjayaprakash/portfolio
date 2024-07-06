variable "region" {
  type = string
  default = "us-east1"
}
variable "project_id" {
  type = string
  default = "complete-will-428105-h5"
}
variable "state_bucket" {
  type = string
  default = "portfolio_buckets_2024"
}
variable "cluster_name" {
  type = string
  default = "churn-prediction-cluster"
}
variable "service_name" {
  type = string
  default = "churn-prediction-cluster"
}
variable "k8s_version" {
  type = string
  default = "1.29.4-gke.1043002"
}
variable "min_node_count" {
  type = string
  default = "1"
}

variable "max_node_count" {
  type = string
  default = "4"
}

variable "machine_type" {
  type = string
  default = "n1-standard-1"
}

variable "preemptible" {
  type = string
  default = "False"
}

