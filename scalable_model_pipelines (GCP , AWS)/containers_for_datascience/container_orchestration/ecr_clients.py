import boto3

def get_clients(service_type,region=None):

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

            # Print repository details
            print("ECR Repositories:")
            for repo in response.get('repositories', []):
                print(f"- Repository Name: {repo['repositoryName']}")
                print(f"  URI: {repo['repositoryUri']}")
                print(f"  Created At: {repo['createdAt']}")
                print()

    except Exception as e:
        raise e


if __name__ == "__main__":
    service_type = 'ecr'
    get_clients(service_type,region="us-west-1")

