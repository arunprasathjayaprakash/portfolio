import os
import subprocess
from google.cloud import artifactregistry_v1
from google.auth import credentials
import docker

# Set GOOGLE_APPLICATION_CREDENTIALS for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "path_to_your_gcp_credentials.json"


def get_gcp_client_and_repos(project_id, location, repository_id, verbose=True):
    """
    Returns GCP Artifact Registry client and the specified repository details.

    Args:
        project_id (str): GCP project ID
        location (str): Region where the repository is located
        repository_id (str): Name of the Artifact Registry repository
        verbose (bool): If True, prints the repository details

    Returns:
        client: Artifact Registry client object
        repo_url (str): Repository URL for Docker image
    """
    try:
        client = artifactregistry_v1.ArtifactRegistryClient()

        # Construct the repository name
        repo_name = f"projects/{project_id}/locations/{location}/repositories/{repository_id}"
        repo = client.get_repository(name=repo_name)

        repo_url = f"{location}-docker.pkg.dev/{project_id}/{repository_id}"

        if verbose:
            print("GCP Artifact Registry Repository Details:")
            print(f"Name: {repo.name}")
            print(f"Format: {repo.format_}")
            print(f"Repository URL: {repo_url}")

        return client, repo_url

    except Exception as e:
        raise Exception(f"Failed to get GCP repositories: {e}")


def create_docker_image(image_tag, docker_config_path):
    """
    Builds a Docker image using the specified tag and Dockerfile path.

    Args:
        image_tag (str): The tag for the Docker image
        docker_config_path (str): Path to the Docker application/configuration

    Returns:
        None
    """
    try:
        client = docker.from_env()
        print("Building Docker Image...")

        image, logs = client.images.build(
            path=docker_config_path,
            tag=image_tag
        )

        for log in logs:
            if 'stream' in log:
                print(log['stream'], end="")
        print("Docker Image Built Successfully!")

    except Exception as e:
        raise Exception(f"Failed to build Docker image: {e}")


def push_docker_image(repo_url, image_tag):
    """
    Pushes the Docker image to the GCP Artifact Registry.

    Args:
        repo_url (str): Repository URL in Artifact Registry
        image_tag (str): Image tag to push

    Returns:
        None
    """
    try:
        print("Authenticating Docker with GCP Artifact Registry...")
        # Authenticate Docker with GCP Artifact Registry
        subprocess.run(["gcloud", "auth", "configure-docker", "--quiet"], check=True)

        # Tag the Docker image
        full_image_tag = f"{repo_url}:{image_tag}"
        print(f"Tagging image as {full_image_tag}")
        client = docker.from_env()
        client.images.get(image_tag).tag(full_image_tag)

        # Push the Docker image
        print("Pushing Docker Image...")
        push_logs = client.images.push(repository=repo_url, tag=image_tag, stream=True)
        for log in push_logs:
            print(log)

        print("Docker Image Pushed Successfully!")

    except Exception as e:
        raise Exception(f"Failed to push Docker image: {e}")


if __name__ == "__main__":
    # Configuration Parameters
    PROJECT_ID = "your-gcp-project-id"
    LOCATION = "us-central1"
    REPOSITORY_ID = "your-repository-id"
    IMAGE_TAG = "gcp-docker-example:1.0"
    DOCKER_CONFIG_PATH = "./app"

    client, repo_url = get_gcp_client_and_repos(PROJECT_ID, LOCATION, REPOSITORY_ID)
    create_docker_image(IMAGE_TAG, DOCKER_CONFIG_PATH)
    push_docker_image(repo_url, IMAGE_TAG)
