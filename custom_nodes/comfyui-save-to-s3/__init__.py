import os
import json
import time
import gc
import numpy as np
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args

# Load environment variables from .env file at ComfyUI root level (like Node.js dotenv)
try:
    from dotenv import load_dotenv
    # Get ComfyUI root directory (where folder_paths.py is located)
    comfyui_root = os.path.dirname(os.path.realpath(folder_paths.__file__))
    # Also check parent directory (runpod-slim level)
    parent_dir = os.path.dirname(comfyui_root)
    
    # Try loading .env from ComfyUI root first, then parent directory
    env_loaded = load_dotenv(os.path.join(comfyui_root, '.env'))
    if not env_loaded:
        load_dotenv(os.path.join(parent_dir, '.env'))
    
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    # Environment variables will still work from system/env, just not from .env file
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

from custom_nodes.s3_client_helper import (
    BOTO3_AVAILABLE,
    ClientError,
    create_s3_client,
)

if BOTO3_AVAILABLE:
    from boto3.s3.transfer import TransferConfig
else:
    TransferConfig = None

try:
    from comfy_api.latest import io, ComfyExtension
    from comfy_api.input import VideoInput, ImageInput
    from comfy_api.util import VideoContainer, VideoCodec
    from typing_extensions import override
    COMFY_API_AVAILABLE = True
except ImportError:
    COMFY_API_AVAILABLE = False
    print("Warning: comfy_api not available. This node will not be loaded.")

# Multipart upload threshold (5MB) - files larger than this use multipart upload
MULTIPART_THRESHOLD = 5 * 1024 * 1024  # 5MB
MULTIPART_CHUNKSIZE = 5 * 1024 * 1024  # 5MB chunks

# Only define the node class if comfy_api is available
if COMFY_API_AVAILABLE:
    class SaveToS3(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="SaveToS3",
                display_name="Save to S3",
                category="image/video",
                description="Memory-efficient S3 upload for images (PNG) and videos (MP4). Automatically detects input type and uses shared AWS credential resolution.",
                inputs=[
                    io.Image.Input(
                        "images",
                        optional=True,
                        tooltip="Images to save to S3 (PNG format)."
                    ),
                    io.Video.Input(
                        "video",
                        optional=True,
                        tooltip="Video to save to S3 (MP4 format)."
                    ),
                    io.String.Input(
                        "filename_prefix",
                        default="ComfyUI",
                        tooltip="The prefix for the file to save."
                    ),
                    io.String.Input(
                        "s3_bucket",
                        default="",
                        tooltip="S3 bucket name (leave empty to use AWS_BUCKET_CONTENT env var)"
                    ),
                    io.Combo.Input(
                        "video_format",
                        options=VideoContainer.as_input(),
                        default="auto",
                        tooltip="The video format/container (mp4, mov, etc.) - only used for video inputs"
                    ),
                    io.Combo.Input(
                        "video_codec",
                        options=VideoCodec.as_input(),
                        default="auto",
                        tooltip="The video codec (h264, etc.) - only used for video inputs"
                    ),
                ],
                outputs=[],
                hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, images=None, video=None, filename_prefix="ComfyUI", s3_bucket="", video_format="auto", video_codec="auto") -> io.NodeOutput:
            # Detect input type
            if images is not None:
                return cls._save_images_to_s3(cls, images, filename_prefix, s3_bucket)
            elif video is not None:
                return cls._save_video_to_s3(cls, video, filename_prefix, s3_bucket, video_format, video_codec)
            else:
                raise ValueError("Either 'images' or 'video' input must be provided")

        @staticmethod
        def _save_images_to_s3(cls, images, filename_prefix, s3_bucket):
            """Memory-efficient image saving to S3."""
            # Use provided bucket or fall back to environment variable
            bucket = s3_bucket or os.environ.get('AWS_BUCKET_CONTENT', '')
            
            if not bucket:
                raise ValueError("S3 bucket name must be provided either as input parameter or via AWS_BUCKET_CONTENT environment variable")
            
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is not installed. Install it with: pip install boto3")
            
            # Initialize S3 client and transfer config
            region_name = os.environ.get('AWS_REGION', 'us-west-2')
            try:
                # Transfer config for multipart uploads
                transfer_config = TransferConfig(
                    multipart_threshold=MULTIPART_THRESHOLD,
                    max_concurrency=10,
                    multipart_chunksize=MULTIPART_CHUNKSIZE,
                )

                s3_client = create_s3_client(region=region_name)
            except Exception as e:
                raise RuntimeError(f"Could not initialize S3 client: {e}")
            
            # Format filename prefix with date/time variables
            prefix = ''
            now = time.localtime()
            filename_prefix = filename_prefix.replace("%year%", str(now.tm_year))
            filename_prefix = filename_prefix.replace("%month%", str(now.tm_mon).zfill(2))
            filename_prefix = filename_prefix.replace("%day%", str(now.tm_mday).zfill(2))
            filename_prefix = filename_prefix.replace("%hour%", str(now.tm_hour).zfill(2))
            filename_prefix = filename_prefix.replace("%minute%", str(now.tm_min).zfill(2))
            filename_prefix = filename_prefix.replace("%second%", str(now.tm_sec).zfill(2))
            
            # Get dimensions once
            if len(images) > 0:
                width = images[0].shape[1]
                height = images[0].shape[0]
                filename_prefix = filename_prefix.replace("%width%", str(width))
                filename_prefix = filename_prefix.replace("%height%", str(height))
            
            results = []
            counter = 0
            
            # Get prompt and extra_pnginfo from hidden inputs
            prompt = cls.hidden.prompt if hasattr(cls, 'hidden') else None
            extra_pnginfo = cls.hidden.extra_pnginfo if hasattr(cls, 'hidden') else None
            
            # Process images one at a time to minimize memory usage
            for batch_number, image in enumerate(images):
                img_buffer = None
                img = None
                img_array = None
                
                try:
                    # Convert tensor to numpy array (move to CPU first)
                    image_cpu = image.cpu()
                    img_array = (255. * image_cpu.numpy()).astype(np.uint8)
                    img_array = np.clip(img_array, 0, 255)
                    
                    # Free tensor memory immediately
                    del image_cpu, image
                    gc.collect()
                    
                    # Convert to PIL Image
                    img = Image.fromarray(img_array)
                    
                    # Free numpy array memory
                    del img_array
                    gc.collect()
                    
                    # Add metadata
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    
                    # Generate filename with counter
                    filename_with_batch = filename_prefix.replace("%batch_num%", str(batch_number))
                    s3_key = f"{filename_with_batch}_{counter:05d}.png"
                    
                    # Save to BytesIO (streaming, memory efficient)
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG', pnginfo=metadata, compress_level=4, optimize=True)
                    img_buffer.seek(0)
                    
                    # Free PIL image memory
                    del img
                    gc.collect()
                    
                    # Upload to S3 (uses multipart if buffer is large enough)
                    s3_client.upload_fileobj(
                        img_buffer,
                        bucket,
                        s3_key,
                        Config=transfer_config
                    )
                    
                    # Generate URLs
                    s3_url = f"s3://{bucket}/{s3_key}"
                    
                    # Generate presigned URL
                    try:
                        public_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': bucket, 'Key': s3_key},
                            ExpiresIn=3600
                        )
                    except:
                        public_url = f"https://{bucket}.s3.{region_name}.amazonaws.com/{s3_key}"
                    
                    results.append({
                        "filename": s3_key.split('/')[-1],
                        "subfolder": "",
                        "type": "output",
                        "s3_url": s3_url,
                        "public_url": public_url
                    })

                    print(f"Successfully uploaded image to S3: {s3_url}")
                    print(f"Public URL (expires in 1h): {public_url}")
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    error_msg = e.response.get('Error', {}).get('Message', str(e))
                    raise RuntimeError(f"Failed to upload image to S3: {error_code} - {error_msg}")
                except Exception as e:
                    raise RuntimeError(f"Error uploading image to S3: {e}")
                finally:
                    # Explicitly free memory
                    if img_buffer is not None:
                        img_buffer.close()
                        del img_buffer
                    if img is not None:
                        del img
                    if img_array is not None:
                        del img_array
                    gc.collect()
                
                counter += 1

            # Return empty output (this is an output node that saves to S3)
            return io.NodeOutput()

        @staticmethod
        def _save_video_to_s3(cls, video: VideoInput, filename_prefix, s3_bucket, video_format, video_codec):
            """Memory-efficient video saving to S3 using streaming encoding."""
            # Use provided bucket or fall back to environment variable
            bucket = s3_bucket or os.environ.get('AWS_BUCKET_CONTENT', '')
            region_name = os.environ.get('AWS_REGION', 'us-west-2')
            
            if not bucket:
                raise ValueError("S3 bucket name must be provided either as input parameter or via AWS_BUCKET_CONTENT environment variable")
            
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is not installed. Install it with: pip install boto3")
            
            # Initialize S3 client and transfer config
            try:
                # Transfer config for multipart uploads
                transfer_config = TransferConfig(
                    multipart_threshold=MULTIPART_THRESHOLD,
                    max_concurrency=10,
                    multipart_chunksize=MULTIPART_CHUNKSIZE,
                )

                s3_client = create_s3_client(region=region_name)
            except Exception as e:
                raise RuntimeError(f"Could not initialize S3 client: {e}")
            
            # Get video dimensions
            width, height = video.get_dimensions()
            
            # Format filename prefix
            prefix = ''
            now = time.localtime()
            filename_prefix = filename_prefix.replace("%year%", str(now.tm_year))
            filename_prefix = filename_prefix.replace("%month%", str(now.tm_mon).zfill(2))
            filename_prefix = filename_prefix.replace("%day%", str(now.tm_mday).zfill(2))
            filename_prefix = filename_prefix.replace("%hour%", str(now.tm_hour).zfill(2))
            filename_prefix = filename_prefix.replace("%minute%", str(now.tm_min).zfill(2))
            filename_prefix = filename_prefix.replace("%second%", str(now.tm_sec).zfill(2))
            filename_prefix = filename_prefix.replace("%width%", str(width))
            filename_prefix = filename_prefix.replace("%height%", str(height))
            
            # Get container format and extension
            container_format = VideoContainer.get_value(video_format) if video_format != "auto" else VideoContainer.MP4
            extension = VideoContainer.get_extension(container_format)
            
            # Generate filename
            counter = 0
            s3_key = f"{filename_prefix}_{counter:05d}.{extension}"
            
            # Get video components
            components = video.get_components()
            images = components.images
            frame_rate = components.frame_rate
            audio = components.audio
            
            # Use streaming encoding with BytesIO to minimize memory
            video_buffer = BytesIO()
            
            try:
                import av
                from fractions import Fraction
                
                # Encode video to MP4 using PyAV with streaming
                with av.open(video_buffer, mode='w', format='mp4', options={'movflags': 'use_metadata_tags'}) as container:
                    # Add metadata if available
                    if not args.disable_metadata:
                        if cls.hidden.prompt is not None:
                            container.metadata['prompt'] = json.dumps(cls.hidden.prompt)
                        if cls.hidden.extra_pnginfo is not None:
                            for x in cls.hidden.extra_pnginfo:
                                container.metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])
                    
                    # Create video stream
                    stream = container.add_stream('h264', rate=Fraction(round(frame_rate * 1000), 1000))
                    stream.width = images.shape[2]
                    stream.height = images.shape[1]
                    stream.pix_fmt = 'yuv420p'
                    
                    # Create audio stream if audio exists
                    audio_stream = None
                    if audio is not None:
                        audio_sample_rate = int(audio['sample_rate'])
                        audio_stream = container.add_stream('aac', rate=audio_sample_rate)
                    
                    # Encode frames one at a time (streaming)
                    for frame_idx, frame_tensor in enumerate(images):
                        # Process frame: convert to numpy, then to VideoFrame
                        frame_cpu = frame_tensor.cpu()
                        frame_array = (frame_cpu * 255).clamp(0, 255).byte().numpy()
                        
                        # Free tensor memory immediately
                        del frame_cpu, frame_tensor
                        
                        # Convert to VideoFrame
                        frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                        frame = frame.reformat(format='yuv420p')
                        
                        # Free numpy array
                        del frame_array
                        gc.collect()
                        
                        # Encode and mux frame
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        
                        del frame
                        
                        # Periodic garbage collection every 30 frames
                        if frame_idx % 30 == 0:
                            gc.collect()
                    
                    # Flush video stream
                    for packet in stream.encode():
                        container.mux(packet)
                    
                    # Note: Audio encoding would go here if needed
                    # For now, we skip audio to save memory
                
                video_buffer.seek(0)

                # Upload to S3 (automatically uses multipart for large files)
                s3_client.upload_fileobj(
                    video_buffer,
                    bucket,
                    s3_key,
                    Config=transfer_config
                )
                
                # Generate URLs
                s3_url = f"s3://{bucket}/{s3_key}"
                
                # Generate presigned URL
                try:
                    public_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket, 'Key': s3_key},
                        ExpiresIn=3600
                    )
                except:
                    public_url = f"https://{bucket}.s3.{region_name}.amazonaws.com/{s3_key}"
                
                print(f"Successfully uploaded video to S3: {s3_url}")
                print(f"Public URL (expires in 1h): {public_url}")

                # Return empty output (this is an output node that saves to S3)
                return io.NodeOutput()
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_msg = e.response.get('Error', {}).get('Message', str(e))
                raise RuntimeError(f"Failed to upload video to S3: {error_code} - {error_msg}")
            except Exception as e:
                raise RuntimeError(f"Error saving video to S3: {e}")
            finally:
                # Explicitly free memory
                if video_buffer is not None:
                    video_buffer.close()
                    del video_buffer
                gc.collect()

if COMFY_API_AVAILABLE:
    class SaveToS3Extension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [
                SaveToS3,
            ]

    async def comfy_entrypoint() -> SaveToS3Extension:
        return SaveToS3Extension()
