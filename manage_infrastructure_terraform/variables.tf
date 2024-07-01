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
  default = "portfolio_proejct_files"
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

#region = "us-east1"
#project_id = "oceanic-beach-419220"
#state_bucket = "credentials_2024_educative"
#cluster_name = "churn-prediction-cluster"
#service_name = "churn-prediction-cluster"
#k8s_version = "1.29.4-gke.1043002"
#min_node_count = "1"
#max_node_count = "4"
#machine_type = "n1-standard-1"
#preemptible = "False"