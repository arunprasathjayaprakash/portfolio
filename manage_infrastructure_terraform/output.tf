output "url" {
  value = "${google_cloud_run_service.churn-docker.status[0].url}"
}