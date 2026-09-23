# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
import shutil
import cv2

from core.bundle_paths import get_data_dir

class StorageManager:
    """
    Centralized storage manager for Recall.
    Enforces a strict lifecycle for temp, cache, and output directories.
    """
    def __init__(self, 
                 delete_temp: bool = True, 
                 debug_frames: bool = False,
                 debug_vision: bool = False,
                 debug_ocr: bool = False):
                 
        self.base_dir = get_data_dir()
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.outputs_dir = os.path.join(self.base_dir, "exports") # keep exports naming for compatibility
        self.assets_dir = os.path.join(self.base_dir, "assets")
        
        # Settings
        self.delete_temp = delete_temp
        self.debug_frames = debug_frames
        self.debug_vision = debug_vision
        self.debug_ocr = debug_ocr
        
    def init_workspace(self):
        """Ensure all base directories exist."""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)
        
        # Make specific debug dirs if needed
        if self.debug_frames:
            os.makedirs(os.path.join(self.temp_dir, "frames"), exist_ok=True)
            
    def cleanup_temp(self):
        """Purge the temp directory if deletion is enabled."""
        if self.delete_temp and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory: {e}")
                
    # --- Debug Helpers ---
    def save_debug_frame(self, frame_img, filename: str):
        """Save a frame to disk only if debug mode is active."""
        if self.debug_frames:
            filepath = os.path.join(self.temp_dir, "frames", filename)
            cv2.imwrite(filepath, frame_img)
