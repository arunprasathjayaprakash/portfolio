import os
import boto3

def get_clients(service_type,region=None,verbose=True):
    '''Returns aws client with selected service , repositories

    args: Service type (ECR, S3 , ECS etc.) , region
    returns: client object , repo list
    '''

    try:
        if region != None:
            region = region

        import json
        creds = json.load(open('credentials.json', 'rb'))
        aws_access_key_id = creds['aws_access_key_id']
        aws_secret_access_key = creds['aws_secret_access_key']

        location = {'LocationConstraint': region}
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )

        client_aws = session.client(service_type,
                                    aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key, region_name=region)

        if service_type == 'ecr':
            response = client_aws.describe_repositories()
            repositories = []

            # Print repository details

            print("ECR Repositories:")
            for repo in response.get('repositories', []):
                if verbose:
                    print(f"- Repository Name: {repo['repositoryName']}")
                    print(f"  URI: {repo['repositoryUri']}")
                    print(f"  Created At: {repo['createdAt']}")
                repositories.append({repo['repositoryName']: repo['repositoryUri']})

            if repositories:
                return client_aws , repositories

    except Exception as e:
        raise e

def create_docker_image(image_tag,docker_config_path):
    '''Creates docker image with specified image tag and docker

    args: image tag , docker application path
    returns: None
    '''

    import docker

    client = docker.from_env()

    image , log = client.images.build(
        path=docker_config_path,
        tag=image_tag
    )

    for log in log:
        if 'stream' in log:
            print(log['stream'], end="")


def push_docker_image(repo_url,image_tag):
    '''Pushes image to cloud repository (AWS)

    args: repository URL , Image tag
    returns: None
    '''

    import docker , base64

    docker_client = docker.from_env()

    ecr_client , repositories = get_clients(service_type,region="us-west-1",verbose=False)

    ecr_credentials = (
        ecr_client
        .get_authorization_token()
        ['authorizationData'][0])

    ecr_username = 'AWS'

    ecr_password = (
        base64.b64decode(ecr_credentials['authorizationToken'])
        .replace(b'AWS:', b'')
        .decode('utf-8'))

    ecr_url = ecr_credentials['proxyEndpoint']

    # get Docker to login/authenticate with ECR
    docker_client.login(
        username=ecr_username, password=ecr_password, registry=ecr_url)

    log = docker_client.images.push(
        repository=repo_url,
        tag=image_tag,
        stream=True
    )

    for l in log:
        print(l)

if __name__ == "__main__":
    service_type = 'ecr'
    client , repositories = get_clients(service_type,region="us-west-1")
    repo_uri = repositories[0]['educative/containers']
    create_docker_image(repo_uri+":1.0","./app")
    push_docker_image(repo_uri,"1.0")
