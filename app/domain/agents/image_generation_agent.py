import io
from app.api.exceptions import APIError
from typing import Optional, List
from google import genai
from google.genai.types import GenerateContentConfig
from app.config import settings
from PIL import Image
from app.infrastructure.blob.azure_media_uploader import AzureBlobUploader
from app.domain.tools.image_downloader import download_image_from_url
from app.domain.tools.gemini_image_parser import parse_image_response

# from io import BytesIO


class ImageGenerationAgent:
    """
    Agent for generating images using the direct Gemini SDK.
    - For text-to-image: pass only a prompt.
    - For image-to-image: pass a prompt and a list of local image paths.
    Returns image bytes (PNG) or None if not found.
    """

    llm = genai.Client(api_key=settings.GEMINI_API_KEY)

    def __init__(self):
        self.prompt = """
You are an AI assistant that ONLY generates images.

- If the user mixes an image request with other types of requests (such as text, captions, or code), respond politely and guide them to separate their queries. 
  Use a helpful and professional tone, not restrictive. 
  Example tone (do not repeat exactly, vary wording naturally):  
    - "It looks like your request included both image and text. For the best results, please make text queries separately. Here’s the image result for your request:"  
    - "I noticed you combined text and image in one request. You can submit text queries separately, but here is the generated image for your request:"  
    - "Your query included both text and image. You may want to ask text requests separately. In the meantime, here’s your image result:"  

- If the user provides a valid image request only, respond with a short professional acknowledgement before the image. 
  Example variations:  
    - "Here is the image you requested:"  
    - "This is the generated image based on your description:"  
    - "Here’s your image result:"  

- After this short message, always output the image inline_data.  
- Never output unrelated text (e.g., captions, social media content, code) together with the image response.  

If the user request contains NO valid image request, respond with:  
  "You can only submit requests for image generation. Please revise your query accordingly."
"""

    async def generate_image(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
    ) -> Optional[bytes]:
        """
        Generate an image using a text prompt, with optional image URLs as input.

        Args:
            prompt: The text prompt for the model.
            image_urls: Optional list of image URLs to download and use as model input.

        Returns:
            dict: {
                "text": str (model's response/acknowledgement for the image prompt),
                "images": List[str] (Azure Blob Storage URLs for the generated images)
            }

        Raises:
            APIError: If generation or upload fails.
        """
        try:
            if not prompt or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")
            # Create the chat object (do not await)
            chat = self.llm.aio.chats.create(
                model="gemini-2.5-flash-image-preview",
                config=GenerateContentConfig(
                    system_instruction=[self.prompt],
                    # 'Multiple candidates is not enabled for models/gemini-2.5-flash-image-preview'
                    # candidate_count=3
                ),
            )
            message = [prompt]

            # Process images from URLs
            if image_urls:
                for url in image_urls:
                    img_bytes = await download_image_from_url(url)
                    img = Image.open(io.BytesIO(img_bytes))
                    message.append(img)

            response = await chat.send_message(
                message=message,
            )
        except Exception as e:
            raise APIError(
                str(e),
                status_code=422,
                public_message=f"Failed to generate image",
            )
        try:
            parsed_response = parse_image_response(response)
            # Upload images to Azure Blob Storage and get URLs
            uploader = AzureBlobUploader()
            image_urls = []
            for img_b64 in parsed_response.get("images", []):
                url = uploader.upload_base64_image(img_b64, file_ext="webp")
                image_urls.append(url)
            return {
                "text": parsed_response.get("text", ""),
                "images": image_urls,
            }
        except Exception as e:
            raise APIError(
                str(e),
                status_code=422,
                public_message=f"Failed to parse image generation response",
            )
