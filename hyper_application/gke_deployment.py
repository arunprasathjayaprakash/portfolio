import os
from google.cloud import container_v1
from kubernetes import client, config
import subprocess

# Set Google Application Credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "<Path to credential file>"

def create_gke_cluster(project_id, region, cluster_name):
    """
    Creates a GKE cluster using Google Cloud client libraries.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Region to create the cluster.
        cluster_name (str): Name of the GKE cluster.

    Returns:
        None
    """
    try:
        # Initialize client
        client_gke = container_v1.ClusterManagerClient()

        # Configure cluster settings
        cluster = {
            "name": cluster_name,
            "initial_node_count": 2,
            "node_config": {
                "machine_type": "e2-medium",
                "disk_size_gb": 100
            },
        }

        print(f"Creating GKE Cluster: {cluster_name} in {region}...")
        parent = f"projects/{project_id}/locations/{region}"
        response = client_gke.create_cluster(request={"parent": parent, "cluster": cluster})
        print("Cluster creation initiated. This may take a few minutes.")

        print("Wating for cluster creation to complete.....")
        import time
        time.sleep(350)
    except Exception as e:
        print(f"{e}")

def get_kubernetes_credentials(project_id, cluster_name, region):
    """
    Retrieves and configures Kubernetes credentials for a GKE cluster.

    Args:
        project_id (str): Google Cloud project ID.
        cluster_name (str): Name of the GKE cluster.
        region (str): Region where the cluster is deployed.

    Returns:
        None
    """
    try:
        # Use gcloud CLI internally to fetch credentials (workaround for kubeconfig)
        import google.auth
        from google.cloud.container_v1 import ClusterManagerClient

        print("Fetching Kubernetes credentials for the GKE cluster...")
        credentials, project = google.auth.default()
        client_gke = ClusterManagerClient(credentials=credentials)
        parent = f"projects/{project_id}/locations/{region}/clusters/{cluster_name}"

        # Generate kubeconfig automatically
        os.system(f"gcloud container clusters get-credentials {cluster_name} --region {region}")
        print("Kubernetes credentials configured successfully.")
    except Exception as e:
        print(f"Failed to configure Kubernetes credentials: {e}")

def get_client():
    import google.auth
    from google.cloud.container_v1 import ClusterManagerClient

    credentials, project = google.auth.default()
    client_gke = ClusterManagerClient(credentials=credentials)

    return credentials,project,client_gke

def deploy_to_gke(image_url,
                  deployment_name,
                  service_name,
                  port,
                  target_port,create_cluster=True):

    """
    Deploys a containerized application to the GKE cluster.

    Args:
        image_url (str): Container image URL in Artifact Registry.
        deployment_name (str): Kubernetes deployment name.
        service_name (str): Kubernetes service name.
        port (int): Port to expose the application.

    Returns:
        None
    """
    try:
        credentials, project, gke_client = get_client()

        if create_cluster:
            create_gke_cluster(credentials.project_id,
                            'us-central1-a',
                            'portfolio')

            # Initialize the GKE client
            gke_client = container_v1.ClusterManagerClient()

            # Fetch cluster details
            location = f"projects/{credentials.project_id}/locations/us-central1-f"
            cluster = gke_client.get_cluster(name=f"{location}/clusters/{deployment_name}")

            # Define the kubeconfig structure
            kubeconfig = {
                "apiVersion": "v1",
                "kind": "Config",
                "clusters": [
                    {
                        "name": cluster.name,
                        "cluster": {
                            "certificate-authority-data": cluster.master_auth.cluster_ca_certificate,
                            "server": f"https://{cluster.endpoint}"  # Change server directly here
                        }
                    }
                ],
                "contexts": [
                    {
                        "name": cluster.name,
                        "context": {
                            "cluster": cluster.name,
                            "user": cluster.name
                        }
                    }
                ],
                "current-context": cluster.name,
                "users": [
                    {
                        "name": cluster.name,
                        "user": {
                            "auth-provider": {
                                "name": "gcp"
                            }
                        }
                    }
                ]
            }

            import yaml
            kubeconfig_path = os.path.expanduser("~/.kube/config")
            os.makedirs(os.path.dirname(kubeconfig_path), exist_ok=True)
            with open(kubeconfig_path, "w") as f:
                yaml.dump(kubeconfig, f, default_flow_style=False)

            print(f"Kubeconfig for cluster '{cluster.name,}' has been updated at {kubeconfig_path}.")

        config.load_config()

        # Kubernetes API clients
        apps_v1_api = client.AppsV1Api()
        core_v1_api = client.CoreV1Api()

        # Define the Deployment
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=client.V1DeploymentSpec(
                replicas=1, #keeping it minimmum for small scale deployments
                selector={"matchLabels": {"app": deployment_name}},
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
                    spec=client.V1PodSpec(containers=[
                        client.V1Container(
                            name=deployment_name,
                            image=image_url,
                            ports=[client.V1ContainerPort(container_port=port)]
                        )
                    ])
                )
            )
        )

        # Create the Deployment
        print("Creating Kubernetes deployment...")
        apps_v1_api.create_namespaced_deployment(namespace="default", body=deployment)
        print(f"Deployment '{deployment_name}' created successfully.")

        # Define the Service
        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=service_name),
            spec=client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[client.V1ServicePort(port=port, target_port=target_port)],
                type="LoadBalancer"
            )
        )

        # Create the Service
        print("Creating Kubernetes service...")
        core_v1_api.create_namespaced_service(namespace="default", body=service)
        print(f"Service '{service_name}' created successfully. It may take some time to provision an external IP.")

    except Exception as e:
        print(f"Error deploying to GKE: {e}")

def update_deployemnt(image,
                      deployment_name,
                      port,
                      existing_name,
                      existing_deployment):

    clus_client = client.AppsV1Api()

    deployment = clus_client.read_namespaced_deployment(
        name = existing_name,
        namespace=existing_deployment
    )

    new_container = client.V1Container(
        name=deployment_name,
        image=image,
        ports=[client.V1ContainerPort(container_port=port)],
    )
    deployment.spec.template.spec.containers.append(new_container)

    api_response = clus_client.patch_namespaced_deployment(
        name=deployment_name, namespace=existing_name, body=deployment
    )

    print(f"Deployment '{DEPLOYMENT_NAME}' updated successfully.")

if __name__ == "__main__":
    IMAGE_URL = "us-central1-docker.pkg.dev/<project_name>/docker-repo/yolo_object_detection:1.0"
    DEPLOYMENT_NAME = "portfolio"
    SERVICE_NAME = "portfolio"
    PORT = 85
    TARGET_PORT = 8505

    deploy_to_gke(IMAGE_URL, DEPLOYMENT_NAME, SERVICE_NAME, PORT,TARGET_PORT,create_cluster=False)

    print("\nDeployment completed. Check the external IP of the service using:")
    print(f"kubectl get service {SERVICE_NAME}")
