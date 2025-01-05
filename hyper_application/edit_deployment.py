from kubernetes import client, config
def update_deployment(image,
                      container_name,
                      port,
                      deployment_ex,
                      namespace_ex):

    config.load_config()

    clus_client = client.AppsV1Api()

    deployment = clus_client.read_namespaced_deployment(
        name = deployment_ex,
        namespace=namespace_ex
    )

    new_container = client.V1Container(
        name=container_name,
        image=image,
        ports=[client.V1ContainerPort(container_port=port)],
    )
    deployment.spec.template.spec.containers.append(new_container)

    api_response = clus_client.patch_namespaced_deployment(
        name=deployment_ex, namespace=namespace_ex, body=deployment
    )

    print(f"Deployment '{deployment_ex}' updated successfully.")


if __name__ == "__main__":
    new_image = "us-central1-docker.pkg.dev/manifest-glyph-441000-g2/docker-repo/self_supervised:1.0"
    new_container = "self-supervised"
    deployment_ex = 'hyper-application'
    namespace_ex = "default"
    new_image_port = 8501
    update_deployment(
        image = new_image,
        port = new_image_port,
        container_name=new_container,
        deployment_ex=deployment_ex,
        namespace_ex=namespace_ex
    )