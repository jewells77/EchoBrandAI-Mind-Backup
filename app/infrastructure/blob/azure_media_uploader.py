import base64
import uuid
import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
from typing import Optional
from app.config import settings


class AzureBlobUploader:
    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        self.connection_string = (
            connection_string or settings.AZURE_BLOB_CONNECTION_STRING
        )
        self.container_name = container_name or settings.AZURE_BLOB_CONTAINER
        self.service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.service_client.get_container_client(
            self.container_name
        )

    def upload_base64_image(
        self, base64_str: str, file_ext: str = "webp", mime_type: Optional[str] = None
    ) -> str:
        img_bytes = base64.b64decode(base64_str)
        date_str = datetime.datetime.now().strftime("%d-%m-%Y")
        blob_name = f"{date_str}-{uuid.uuid4()}.{file_ext}"
        blob_client = self.container_client.get_blob_client(blob_name)

        # Safely determine MIME type
        if mime_type:
            ct = mime_type
        else:
            ct = f"image/{file_ext.lower()}"  # default fallback

        content_settings = ContentSettings(
            content_type=ct, content_disposition="inline"  # 👈 allow in-browser viewing
        )

        blob_client.upload_blob(
            img_bytes,
            blob_type="BlockBlob",
            overwrite=True,
            content_settings=content_settings,
        )

        return blob_client.url

    def upload_video_file(
        self, video_bytes: bytes, file_ext: str = "mp4", mime_type: Optional[str] = None
    ) -> str:
        """
        Upload a video file to Azure Blob Storage.
        Args:
            video_bytes: The video file in bytes.
            file_ext: The file extension (default: mp4).
            mime_type: Optional MIME type (default: 'video/<file_ext>').
        Returns:
            str: URL to the uploaded blob.
        """
        date_str = datetime.datetime.now().strftime("%d-%m-%Y")
        blob_name = f"{date_str}-{uuid.uuid4()}.{file_ext}"
        blob_client = self.container_client.get_blob_client(blob_name)

        # Safely determine MIME type
        if mime_type:
            ct = mime_type
        else:
            ct = f"video/{file_ext.lower()}"  # default fallback

        content_settings = ContentSettings(
            content_type=ct, content_disposition="inline"  # allow in-browser viewing
        )

        blob_client.upload_blob(
            video_bytes,
            blob_type="BlockBlob",
            overwrite=True,
            content_settings=content_settings,
        )

        return blob_client.url
