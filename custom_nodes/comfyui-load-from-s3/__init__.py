import gc
import hashlib
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

from custom_nodes.s3_client_helper import (
    BOTO3_AVAILABLE,
    BotoCoreError,
    ClientError,
    create_s3_client,
)

try:
    from comfy_api.latest import io, ComfyExtension
    from typing_extensions import override

    COMFY_API_AVAILABLE = True
except ImportError:
    COMFY_API_AVAILABLE = False


class LoadImageFromS3:
    """Custom node that loads an image tensor from S3 using bucket/key inputs and shared credential handling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "key": ("STRING", {"default": ""}),
            },
            "optional": {
                "aws_region": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_image"

    def load_image(self, bucket, key, aws_region=""):
        if not bucket:
            raise ValueError("S3 bucket name must be provided.")

        if not key:
            raise ValueError("S3 object key must be provided.")

        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for LoadImageFromS3. Install with `pip install boto3`.")

        try:
            s3_client = create_s3_client(region=aws_region or os.environ.get("AWS_REGION"))
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to initialize S3 client: {exc}") from exc

        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            error_message = exc.response.get("Error", {}).get("Message", str(exc))
            raise RuntimeError(f"Failed to fetch object from S3: {error_code} - {error_message}") from exc
        except BotoCoreError as exc:
            raise RuntimeError(f"Failed to fetch object from S3: {exc}") from exc

        image_bytes = BytesIO()
        try:
            # Stream into BytesIO in manageable chunks
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                image_bytes.write(chunk)
        finally:
            body.close()

        image_bytes.seek(0)

        try:
            pil_image = Image.open(image_bytes)
        except Exception as exc:  # Pillow raises many custom exceptions
            raise RuntimeError(f"Failed to decode image from S3 object: {exc}") from exc

        output_tensors = []
        width = height = None

        try:
            for frame in ImageSequence.Iterator(pil_image):
                frame = ImageOps.exif_transpose(frame)
                if frame.mode == "I":
                    frame = frame.point(lambda i: i * (1 / 255))
                frame = frame.convert("RGB")

                if width is None or height is None:
                    width, height = frame.size

                if frame.size != (width, height):
                    # Skip frames that do not match the first frame dimensions
                    continue

                frame_array = np.array(frame).astype(np.float32) / 255.0
                tensor = torch.from_numpy(frame_array)[None, ...]
                output_tensors.append(tensor)
        finally:
            pil_image.close()
            image_bytes.close()

        if not output_tensors:
            raise RuntimeError("No valid frames were decoded from the S3 object.")

        if len(output_tensors) > 1:
            output_image = torch.cat(output_tensors, dim=0)
        else:
            output_image = output_tensors[0]

        # Free intermediate frames
        del output_tensors
        gc.collect()

        return (output_image,)

    @classmethod
    def IS_CHANGED(cls, bucket, key, aws_region=""):
        if not BOTO3_AVAILABLE or not bucket or not key:
            return hashlib.sha256(f"{bucket}:{key}".encode("utf-8")).hexdigest()

        try:
            s3_client = create_s3_client(region=aws_region or os.environ.get("AWS_REGION"))
            metadata = s3_client.head_object(Bucket=bucket, Key=key)
            etag = metadata.get("ETag", "")
            last_modified = metadata.get("LastModified")
        except Exception:
            # Fall back to a hash of the S3 path if metadata lookup fails
            return hashlib.sha256(f"{bucket}:{key}".encode("utf-8")).hexdigest()

        marker = f"{bucket}:{key}:{etag}:{last_modified}"
        return hashlib.sha256(marker.encode("utf-8")).hexdigest()


NODE_CLASS_MAPPINGS = {
    "LoadImageFromS3": LoadImageFromS3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromS3": "Load Image From S3",
}

if COMFY_API_AVAILABLE:

    class LoadImageFromS3Node(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="LoadImageFromS3",
                display_name="Load Image From S3",
                category="image",
                description="Fetches an image from S3 (bucket/key) and outputs it as a tensor.",
                inputs=[
                    io.String.Input(
                        "bucket",
                        tooltip="S3 bucket name.",
                    ),
                    io.String.Input(
                        "key",
                        tooltip="S3 object key/path.",
                    ),
                    io.String.Input(
                        "aws_region",
                        default="",
                        tooltip="Optional AWS region; defaults to environment or AWS SDK config.",
                    ),
                ],
                outputs=[
                    io.Image.Output("image"),
                ],
                documentation="""
                Downloads the specified S3 object and decodes it into an image tensor.
                Requires AWS credentials to be configured via environment, config file, or IAM role.
                """,
            )

        @classmethod
        def execute(cls, bucket, key, aws_region="") -> io.NodeOutput:
            loader = LoadImageFromS3()
            (image_tensor,) = loader.load_image(bucket, key, aws_region)
            return io.NodeOutput(image=image_tensor)

    class LoadImageFromS3Extension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [LoadImageFromS3Node]

    async def comfy_entrypoint() -> LoadImageFromS3Extension:
        return LoadImageFromS3Extension()

