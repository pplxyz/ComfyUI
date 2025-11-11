"""
Webhook Handler for ComfyUI
Sends webhook notifications when prompts complete
"""

import asyncio
import aiohttp
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WebhookHandler:
    """Handles webhook notifications for completed prompts"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def send_webhook(self, webhook_url: str, prompt_id: str, status: str, outputs: dict, meta: dict = None):
        """
        Send webhook notification
        
        Args:
            webhook_url: URL to send webhook to
            prompt_id: The prompt ID that completed
            status: 'success' or 'error'
            outputs: The output data from the prompt
            meta: Optional metadata
        """
        if not webhook_url:
            return
            
        payload = {
            "prompt_id": prompt_id,
            "status": status,
            "outputs": outputs,
            "meta": meta or {}
        }
        
        try:
            session = await self.get_session()
            async with session.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 200 and response.status < 300:
                    logger.info(f"✅ Webhook sent successfully to {webhook_url} for prompt {prompt_id}")
                else:
                    logger.warning(f"⚠️ Webhook failed with status {response.status}: {await response.text()}")
        except asyncio.TimeoutError:
            logger.error(f"❌ Webhook timeout for {webhook_url}")
        except Exception as e:
            logger.error(f"❌ Webhook error for {webhook_url}: {e}")
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()


# Global webhook handler instance
_webhook_handler = None

def get_webhook_handler():
    """Get the global webhook handler instance"""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = WebhookHandler()
    return _webhook_handler

