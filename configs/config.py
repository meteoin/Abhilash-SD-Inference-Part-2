from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import redis
import cloudinary
import cloudinary.uploader
import cloudinary.api
import dramatiq
from dramatiq.brokers.redis import RedisBroker

# Redis
REDIS_CLIENT_URL='localhost'
REDIS_CLIENT_PORT='6379'

default_credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(account_url="https://sdinference7713743774.blob.core.windows.net/", credential=default_credential)

PROJECT_FOLDER = "/home/azureuser/cloudfiles/code/Stable-Diffusion-and-Aws-Sagemaker"

redis_client = redis.Redis(host=REDIS_CLIENT_URL, port=REDIS_CLIENT_PORT, db=0)

IMAGE_URL = "https://res.cloudinary.com/de2bpm2bc/image/upload/v1721347068"

# Configuration
cloudinary.config(
    cloud_name="de2bpm2bc",
    api_key="825717191529899",
    api_secret="v3EolebjVyY5Pp8Pe0zYNma8rZY",
    secure=True
)

redis_broker = RedisBroker(url="redis://localhost:6379/0")
dramatiq.set_broker(redis_broker)

