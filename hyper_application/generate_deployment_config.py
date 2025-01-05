import yaml
import os
from google.cloud import container_v1

def generate_k8s_yaml_with_cluster_info_directly(
    project_id,
    repository_name,
    image_name,
    tag,
    deployment_name,
    service_name,
    replicas,
    container_port,
    service_port,
    load_balancer_type,
    pvc_name,
    storage_size,
    mount_path,
    cluster_name,
    zone
):
    """
    Generates Kubernetes YAML and kubeconfig with GKE cluster information directly embedded.
    """
    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": pvc_name
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {
                "requests": {
                    "storage": storage_size
                }
            },
            "storageClassName": "standard-rwo"
        }
    }

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app": deployment_name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": deployment_name
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "my-container",
                            "image": f"us-central1-docker.pkg.dev/{project_id}/{repository_name}/{image_name}:{tag}",
                            "ports": [{"containerPort": container_port}],
                            "volumeMounts": [{"name": "my-storage", "mountPath": mount_path}]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "my-storage",
                            "persistentVolumeClaim": {"claimName": pvc_name}
                        }
                    ]
                }
            }
        }
    }

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name},
        "spec": {
            "selector": {"app": deployment_name},
            "ports": [{"protocol": "TCP", "port": service_port, "targetPort": container_port}],
            "type": load_balancer_type
        }
    }

    # Write Deployment, PVC, and Service YAML
    k8s_file = "k8s_deployment_service_with_pvc.yaml"
    with open(k8s_file, "w") as f:
        yaml.dump(pvc, f, default_flow_style=False)
        f.write("---\n")
        yaml.dump(deployment, f, default_flow_style=False)
        f.write("---\n")
        yaml.dump(service, f, default_flow_style=False)

    print(f"Kubernetes configuration YAML generated: {k8s_file}")


if __name__ == "__main__":
    from gke_deployment import get_client
    creds , project , _ = get_client()
    generate_k8s_yaml_with_cluster_info_directly(
        project_id=creds.project_id,
        repository_name="docker-repo",
        image_name="yolo_object_detection",
        tag="2.0",
        deployment_name="hyper-app",
        service_name="hyper-app-service",
        replicas=1,
        container_port=8080,
        service_port=80,
        load_balancer_type="LoadBalancer",
        pvc_name="my-pvc",
        storage_size="10Gi",
        mount_path="/data",
        cluster_name="hyper-cluster",
        zone="us-central1-f"
    )
