#!/usr/bin/env python3
"""
Modern glassmorphism overlay panels with smooth animations.

Features:
- Only shows when a person is being tracked
- Smooth fade + slide animations
- Frosted glass effect with blur
- Clean, modern design
- Speaking indicator above tracked person's head
- Toast notifications for social cues
"""
import json
import os
import time
import math
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter


@dataclass
class ToastNotification:
    """A toast notification to display."""
    icon: str
    message: str
    timestamp: float
    duration: float = 4.0  # seconds to display


class OverlayRenderer:
    """Renders modern glassmorphism overlay panels with animations."""

    # Reference resolution for scaling (1080p)
    REFERENCE_HEIGHT = 1080

    def __init__(self):
        # Base panel config (at reference resolution)
        self._base_panel_width = 380
        self._base_panel_gap = 14
        self._base_corner_radius = 16
        self._base_padding_x = 24
        self._base_padding_y = 20

        # Base font sizes (at reference resolution)
        self._base_label_size = 14
        self._base_text_size = 20
        self._base_emotion_size = 28
        self._base_indicator_size = 16
        self._base_toast_size = 18

        # Cached scaled values
        self._cached_scale = 0
        self._cached_height = 0

        # Animation settings
        self.animation_duration = 0.4  # seconds
        self._visible = False
        self._animation_progress = 0.0  # 0 = hidden, 1 = visible
        self._last_update = time.time()

        # Glassmorphism colors
        self.bg_color = (18, 18, 22)  # Near black
        self.bg_alpha = 0.75
        self.border_color = (255, 255, 255)  # White border
        self.border_alpha = 0.12
        self.blur_radius = 20

        # Text colors (RGB for PIL)
        self.white = (255, 255, 255)
        self.label_color = (255, 255, 255, 140)  # Semi-transparent white
        self.text_color = (255, 255, 255)
        self.dim_color = (180, 180, 180)

        # Speaking indicator colors
        self.speaking_color = (220, 60, 60)  # Red when speaking
        self.your_turn_color = (60, 180, 60)  # Green when your turn

        # Initialize with default scale
        self._update_scale(self.REFERENCE_HEIGHT)

        # State file
        env_path = os.getenv("CONVO_STATE_PATH")
        if env_path:
            self.state_file = Path(env_path).expanduser()
        else:
            home_guess = Path("~/Downloads/Nex/conve_context/latest_state.json").expanduser()
            rel_guess = (
                Path(__file__).resolve().parent.parent.parent
                / "conve_context"
                / "latest_state.json"
            )
            self.state_file = home_guess if home_guess.exists() else rel_guess

        # Cached state
        self._cached_state = None
        self._last_state_check = 0
        self._state_check_interval = 0.1  # Faster polling for responsiveness

        # Toast notifications
        self._active_toasts: List[ToastNotification] = []
        self._last_social_cue_timestamp = 0

        # Speaking indicator animation
        self._speaking_pulse = 0.0

        # Quest mode: use Quest-friendly positioning even for mono videos
        # When True, mono videos will use the larger base_x offset and scaled panel width
        # that stereo mode uses, positioning panels more toward center
        self.force_quest_mode = True

        # Clear any stale state file on init
        self._clear_stale_state()

    def _clear_stale_state(self) -> None:
        """Clear cached state and reset toast notifications."""
        self._cached_state = None
        self._active_toasts = []
        self._last_social_cue_timestamp = 0

    def _update_scale(self, frame_height: int) -> None:
        """Update scaled dimensions based on frame resolution."""
        if frame_height == self._cached_height:
            return  # No change needed

        self._cached_height = frame_height
        self._cached_scale = frame_height / self.REFERENCE_HEIGHT

        # Scale panel dimensions
        self.panel_width = int(self._base_panel_width * self._cached_scale)
        self.panel_gap = int(self._base_panel_gap * self._cached_scale)
        self.corner_radius = int(self._base_corner_radius * self._cached_scale)
        self.padding_x = int(self._base_padding_x * self._cached_scale)
        self.padding_y = int(self._base_padding_y * self._cached_scale)

        # Scale indicator dimensions - MUCH wider to fit text
        self.indicator_width = max(180, int(200 * self._cached_scale))
        self.indicator_height = max(40, int(45 * self._cached_scale))
        self.indicator_radius = max(12, int(14 * self._cached_scale))

        # Scale toast dimensions
        self.toast_height = int(34 * self._cached_scale)
        self.toast_radius = int(12 * self._cached_scale)

        # Scale and reload fonts
        self.label_size = max(10, int(self._base_label_size * self._cached_scale))
        self.text_size = max(12, int(self._base_text_size * self._cached_scale))
        self.emotion_size = max(16, int(self._base_emotion_size * self._cached_scale))
        self.indicator_size = max(10, int(self._base_indicator_size * self._cached_scale))
        self.toast_size = max(12, int(self._base_toast_size * self._cached_scale))

        self._font_label = self._load_font(self.label_size, bold=True)
        self._font_text = self._load_font(self.text_size)
        self._font_emotion = self._load_font(self.emotion_size, bold=True)
        self._font_indicator = self._load_font(self.indicator_size, bold=True)
        self._font_toast = self._load_font(self.toast_size, bold=False)
        self._font_toast_icon = self._load_font(int(self.toast_size * 1.1), bold=False)
        self._font_emoji = self._load_emoji_font(int(self.emotion_size * 1.5))

    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Load a TrueType font with fallbacks."""
        if bold:
            font_names = ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
        else:
            font_names = ["segoeui.ttf", "arial.ttf", "calibri.ttf"]

        font_dirs = ["C:/Windows/Fonts/", "/usr/share/fonts/truetype/", "/System/Library/Fonts/"]

        for font_dir in font_dirs:
            for font_name in font_names:
                font_path = os.path.join(font_dir, font_name)
                if os.path.exists(font_path):
                    try:
                        return ImageFont.truetype(font_path, size)
                    except:
                        continue
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

    def _load_emoji_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load an emoji-capable font."""
        # Windows emoji fonts
        emoji_fonts = [
            "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
            "C:/Windows/Fonts/segoe ui emoji.ttf",
            "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux
        ]
        for font_path in emoji_fonts:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        # Fallback to regular font
        return self._load_font(size, bold=True)

    def _map_emotion(self, raw_emotion: str) -> tuple:
        """Map emotion to display text and emoji.

        Only happy and neutral map to Calm. Other emotions display as-is.
        Uses short words to fit in narrow UI panels.

        Returns:
            tuple: (display_text, emoji)
        """
        # Normalize to lowercase for matching
        emotion_lower = raw_emotion.lower().strip()

        # Only these two emotions map to "Calm"
        if "happy" in emotion_lower or "neutral" in emotion_lower:
            return ("Calm", "😌")

        # All other emotions - use short words to fit UI
        emotion_map = {
            "angry": ("Mad", "😠"),
            "sad": ("Sad", "😢"),
            "fear": ("Tense", "😰"),
            "fearful": ("Tense", "😰"),
            "anxious": ("Tense", "😰"),
            "surprise": ("Shock", "😮"),
            "surprised": ("Shock", "😮"),
            "disgust": ("Gross", "😒"),
            "disgusted": ("Gross", "😒"),
            "contempt": ("Smug", "😏"),
        }

        # Check for matches
        for key, (display, emoji) in emotion_map.items():
            if key in emotion_lower:
                return (display, emoji)

        # Default fallback - show raw emotion if unknown (truncate if too long)
        fallback = raw_emotion.capitalize()
        if len(fallback) > 6:
            fallback = fallback[:5] + "."
        return (fallback, "😐")

    def _load_state(self) -> dict:
        """Load conversation state from file."""
        now = time.time()
        if now - self._last_state_check < self._state_check_interval:
            return self._cached_state or {}

        self._last_state_check = now
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    self._cached_state = json.load(f)
                    return self._cached_state
        except:
            pass
        return self._cached_state or {}

    def _ease_out_cubic(self, t: float) -> float:
        """Smooth ease-out animation curve."""
        return 1 - pow(1 - t, 3)

    def _ease_in_cubic(self, t: float) -> float:
        """Smooth ease-in animation curve."""
        return t * t * t

    def _update_animation(self, should_show: bool) -> float:
        """Update animation state and return current progress."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Clamp dt to reasonable range (handles first frame after long delay)
        dt = min(dt, 0.1)

        # Animation speed
        speed = 1.0 / self.animation_duration

        if should_show:
            self._animation_progress = min(1.0, self._animation_progress + dt * speed)
            return self._ease_out_cubic(self._animation_progress)
        else:
            self._animation_progress = max(0.0, self._animation_progress - dt * speed)
            return self._ease_out_cubic(self._animation_progress)

    def _draw_glassmorphism_panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                                   alpha: float = 1.0, _timings: dict = None) -> np.ndarray:
        """Draw a glassmorphism panel with blur effect."""
        if alpha <= 0:
            return frame

        _t = _timings is not None

        radius = self.corner_radius

        # Create mask for rounded rectangle
        if _t: t0 = time.time()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
        cv2.ellipse(mask, (radius, radius), (radius, radius), 180, 0, 90, 255, -1)
        cv2.ellipse(mask, (w - radius, radius), (radius, radius), 270, 0, 90, 255, -1)
        cv2.ellipse(mask, (radius, h - radius), (radius, radius), 90, 0, 90, 255, -1)
        cv2.ellipse(mask, (w - radius, h - radius), (radius, radius), 0, 0, 90, 255, -1)
        if _t: _timings['gp_mask'] = _timings.get('gp_mask', 0) + (time.time() - t0) * 1000

        # Extract region and apply blur (frosted glass effect)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Adjust mask to actual region size
        mask_x1, mask_y1 = x1 - x, y1 - y
        mask_x2, mask_y2 = mask_x1 + (x2 - x1), mask_y1 + (y2 - y1)
        region_mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]

        # Get region and blur it (frosted glass effect)
        if _t: t0 = time.time()
        region = frame[y1:y2, x1:x2].copy()
        if _t: _timings['gp_copy'] = _timings.get('gp_copy', 0) + (time.time() - t0) * 1000

        # Fast blur: downscale 4x → small blur → upscale (same visual effect, much faster)
        if _t: t0 = time.time()
        rh, rw = region.shape[:2]
        small = cv2.resize(region, (rw // 4, rh // 4), interpolation=cv2.INTER_AREA)
        small_blur = cv2.GaussianBlur(small, (7, 7), 0)
        blurred = cv2.resize(small_blur, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if _t: _timings['gp_blur'] = _timings.get('gp_blur', 0) + (time.time() - t0) * 1000

        # Fast blend: use cv2.addWeighted for dark tint, then apply mask
        if _t: t0 = time.time()

        # Dark tint blend (75% dark, 25% blurred background)
        # Instead of creating full dark_overlay array, use scalar blend
        blended = cv2.addWeighted(blurred, 0.25, np.full_like(blurred, (22, 22, 28), dtype=np.uint8), 0.75, 0)

        # Apply with mask using cv2 for speed
        mask_float = region_mask.astype(np.float32) * (0.88 * alpha / 255.0)
        mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

        # Blend: result = blended * mask + region * (1 - mask)
        result = (blended.astype(np.float32) * mask_3ch + region.astype(np.float32) * (1.0 - mask_3ch))
        frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
        if _t: _timings['gp_blend'] = _timings.get('gp_blend', 0) + (time.time() - t0) * 1000

        # Draw outer border - only copy panel region, not entire frame
        if _t: t0 = time.time()
        border_color = (255, 255, 255)  # White border

        # Create small overlay just for the panel region (with padding for border)
        pad = 2
        bx1, by1 = max(0, x - pad), max(0, y - pad)
        bx2, by2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        border_region = frame[by1:by2, bx1:bx2].copy()

        # Offset for drawing within the region
        ox, oy = x - bx1, y - by1

        # Top edge
        cv2.line(border_region, (ox + radius, oy), (ox + w - radius, oy), border_color, 1)
        # Bottom edge
        cv2.line(border_region, (ox + radius, oy + h), (ox + w - radius, oy + h), border_color, 1)
        # Left edge
        cv2.line(border_region, (ox, oy + radius), (ox, oy + h - radius), border_color, 1)
        # Right edge
        cv2.line(border_region, (ox + w, oy + radius), (ox + w, oy + h - radius), border_color, 1)
        # Corners
        cv2.ellipse(border_region, (ox + radius, oy + radius), (radius, radius), 180, 0, 90, border_color, 1)
        cv2.ellipse(border_region, (ox + w - radius, oy + radius), (radius, radius), 270, 0, 90, border_color, 1)
        cv2.ellipse(border_region, (ox + radius, oy + h - radius), (radius, radius), 90, 0, 90, border_color, 1)
        cv2.ellipse(border_region, (ox + w - radius, oy + h - radius), (radius, radius), 0, 0, 90, border_color, 1)

        # Blend only the region back
        frame[by1:by2, bx1:bx2] = cv2.addWeighted(border_region, 0.15 * alpha, frame[by1:by2, bx1:bx2], 1 - 0.15 * alpha, 0)
        if _t: _timings['gp_border1'] = _timings.get('gp_border1', 0) + (time.time() - t0) * 1000

        # Draw inner highlight border (offset inward by 2 pixels) - region-based
        if _t: t0 = time.time()
        inner_offset = 2
        inner_radius = max(radius - inner_offset, 2)
        ix, iy = x + inner_offset, y + inner_offset
        iw, ih = w - inner_offset * 2, h - inner_offset * 2

        # Reuse same region bounds, just need fresh copy
        inner_region = frame[by1:by2, bx1:bx2].copy()
        iox, ioy = ix - bx1, iy - by1

        # Inner top edge
        cv2.line(inner_region, (iox + inner_radius, ioy), (iox + iw - inner_radius, ioy), border_color, 1)
        # Inner bottom edge
        cv2.line(inner_region, (iox + inner_radius, ioy + ih), (iox + iw - inner_radius, ioy + ih), border_color, 1)
        # Inner left edge
        cv2.line(inner_region, (iox, ioy + inner_radius), (iox, ioy + ih - inner_radius), border_color, 1)
        # Inner right edge
        cv2.line(inner_region, (iox + iw, ioy + inner_radius), (iox + iw, ioy + ih - inner_radius), border_color, 1)
        # Inner corners
        cv2.ellipse(inner_region, (iox + inner_radius, ioy + inner_radius), (inner_radius, inner_radius), 180, 0, 90, border_color, 1)
        cv2.ellipse(inner_region, (iox + iw - inner_radius, ioy + inner_radius), (inner_radius, inner_radius), 270, 0, 90, border_color, 1)
        cv2.ellipse(inner_region, (iox + inner_radius, ioy + ih - inner_radius), (inner_radius, inner_radius), 90, 0, 90, border_color, 1)
        cv2.ellipse(inner_region, (iox + iw - inner_radius, ioy + ih - inner_radius), (inner_radius, inner_radius), 0, 0, 90, border_color, 1)

        frame[by1:by2, bx1:bx2] = cv2.addWeighted(inner_region, 0.08 * alpha, frame[by1:by2, bx1:bx2], 1 - 0.08 * alpha, 0)
        if _t: _timings['gp_border2'] = _timings.get('gp_border2', 0) + (time.time() - t0) * 1000

        return frame

    def _wrap_text(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def _draw_speaking_indicator_single(self, frame: np.ndarray, center_x: int,
                                          head_y: float, is_speaking: bool,
                                          alpha: float, pulse: float) -> np.ndarray:
        """Draw glassmorphism speaking indicator at a specific x position."""
        h, w = frame.shape[:2]

        # Calculate position above head (scaled offset)
        offset = int(50 * self._cached_scale)
        indicator_y = int(head_y * h) - offset
        indicator_y = max(int(35 * self._cached_scale), indicator_y)

        # Use scaled indicator dimensions
        indicator_width = self.indicator_width
        indicator_height = self.indicator_height
        radius = self.indicator_radius

        ind_x = center_x - indicator_width // 2
        ind_y = indicator_y - indicator_height // 2

        # Clamp to bounds
        ind_x = max(5, min(ind_x, w - indicator_width - 5))

        # --- Glassmorphism background ---
        # Extract region for blur effect
        x1, y1 = max(0, ind_x - 2), max(0, ind_y - 2)
        x2, y2 = min(w, ind_x + indicator_width + 2), min(h, ind_y + indicator_height + 2)

        if x2 > x1 and y2 > y1:
            # Create rounded rect mask
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            mask_w, mask_h = x2 - x1, y2 - y1

            # Draw rounded rectangle in mask
            cv2.rectangle(mask, (radius, 0), (mask_w - radius, mask_h), 255, -1)
            cv2.rectangle(mask, (0, radius), (mask_w, mask_h - radius), 255, -1)
            cv2.circle(mask, (radius, radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius, radius), radius, 255, -1)
            cv2.circle(mask, (radius, mask_h - radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius, mask_h - radius), radius, 255, -1)

            # Get region and fast blur (downscale → blur → upscale)
            region = frame[y1:y2, x1:x2].copy()
            rh, rw = region.shape[:2]
            if rw > 8 and rh > 8:
                small = cv2.resize(region, (rw // 4, rh // 4), interpolation=cv2.INTER_AREA)
                small_blur = cv2.GaussianBlur(small, (5, 5), 0)
                blurred = cv2.resize(small_blur, (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                blurred = cv2.GaussianBlur(region, (7, 7), 0)

            # Dark tint blend
            blended = cv2.addWeighted(blurred, 0.25, np.full_like(blurred, (22, 22, 28), dtype=np.uint8), 0.75, 0)

            # Apply with mask
            mask_float = mask.astype(np.float32) * (0.88 * alpha / 255.0)
            mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
            result = blended.astype(np.float32) * mask_3ch + region.astype(np.float32) * (1.0 - mask_3ch)
            frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)

            # Subtle border - only copy indicator region, not entire frame
            border_color = (255, 255, 255)
            border_region = frame[y1:y2, x1:x2].copy()

            # Offset for drawing within the region
            ox, oy = ind_x - x1, ind_y - y1

            # Draw border lines
            cv2.line(border_region, (ox + radius, oy), (ox + indicator_width - radius, oy), border_color, 1)
            cv2.line(border_region, (ox + radius, oy + indicator_height), (ox + indicator_width - radius, oy + indicator_height), border_color, 1)
            cv2.line(border_region, (ox, oy + radius), (ox, oy + indicator_height - radius), border_color, 1)
            cv2.line(border_region, (ox + indicator_width, oy + radius), (ox + indicator_width, oy + indicator_height - radius), border_color, 1)

            # Corner arcs
            cv2.ellipse(border_region, (ox + radius, oy + radius), (radius, radius), 180, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + indicator_width - radius, oy + radius), (radius, radius), 270, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + radius, oy + indicator_height - radius), (radius, radius), 90, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + indicator_width - radius, oy + indicator_height - radius), (radius, radius), 0, 0, 90, border_color, 1)

            frame[y1:y2, x1:x2] = cv2.addWeighted(border_region, 0.15 * alpha, frame[y1:y2, x1:x2], 1 - 0.15 * alpha, 0)

        # --- Calculate centered positions first ---
        dot_size = max(8, int(10 * self._cached_scale))
        dot_y = indicator_y
        text_gap = max(8, int(10 * self._cached_scale))

        # Determine text
        text = "SPEAKING" if is_speaking else "YOUR TURN"

        # Measure text to calculate centered layout
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=self._font_indicator)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Calculate total content width (dot + gap + text) and center it
        content_width = (dot_size * 2) + text_gap + text_w
        content_start_x = ind_x + (indicator_width - content_width) // 2

        # Centered dot position
        dot_x = content_start_x + dot_size  # dot center

        # --- Now draw the dot at centered position ---
        if is_speaking:
            # Red/coral accent with pulsing glow
            glow_radius = int(dot_size * 2 + 3 * pulse)
            glow_color = (100, 120, 255)  # BGR - soft coral/red
            dot_color = (130, 140, 255)   # BGR - brighter coral

            # Draw glow - only copy glow region, not entire frame
            gr = glow_radius + 2
            gx1, gy1 = max(0, dot_x - gr), max(0, dot_y - gr)
            gx2, gy2 = min(w, dot_x + gr), min(h, dot_y + gr)
            if gx2 > gx1 and gy2 > gy1:
                glow_region = frame[gy1:gy2, gx1:gx2].copy()
                cv2.circle(glow_region, (dot_x - gx1, dot_y - gy1), glow_radius, glow_color, -1)
                frame[gy1:gy2, gx1:gx2] = cv2.addWeighted(glow_region, 0.3 * alpha, frame[gy1:gy2, gx1:gx2], 1 - 0.3 * alpha, 0)

            # Draw dot
            cv2.circle(frame, (dot_x, dot_y), dot_size, dot_color, -1)
            cv2.circle(frame, (dot_x, dot_y), dot_size, (180, 190, 255), 1)
        else:
            # Green/mint accent
            dot_color = (180, 220, 140)  # BGR - soft mint green
            cv2.circle(frame, (dot_x, dot_y), dot_size, dot_color, -1)
            cv2.circle(frame, (dot_x, dot_y), dot_size, (200, 240, 180), 1)

        # --- Text with PIL - only convert indicator region, not entire frame ---
        # Calculate text region bounds
        text_x = dot_x + dot_size + text_gap - bbox[0]
        text_y = dot_y - text_h // 2 - bbox[1]

        # Region for PIL text rendering (indicator bounds with some padding)
        pad = 5
        tx1, ty1 = max(0, ind_x - pad), max(0, ind_y - pad)
        tx2, ty2 = min(w, ind_x + indicator_width + pad), min(h, ind_y + indicator_height + pad)

        if tx2 > tx1 and ty2 > ty1:
            text_region = frame[ty1:ty2, tx1:tx2]
            text_region_rgb = cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(text_region_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Adjust text position for region offset
            local_text_x = text_x - tx1
            local_text_y = text_y - ty1

            # Draw text
            text_color = (255, 255, 255, int(230 * alpha))
            draw.text((local_text_x, local_text_y), text, font=self._font_indicator, fill=text_color)

            frame[ty1:ty2, tx1:tx2] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return frame

    def _draw_speaking_indicator(self, frame: np.ndarray, head_x: float, head_y: float,
                                   is_speaking: bool, alpha: float = 1.0) -> np.ndarray:
        """Draw speaking indicator above the tracked person's head.

        Handles stereo mode by drawing on both left and right halves.

        Args:
            frame: Video frame to draw on (may be stereo side-by-side)
            head_x: Normalized x position (0-1) of head center
            head_y: Normalized y position (0-1) of head top
            is_speaking: Whether the other person is speaking
            alpha: Animation alpha for fade effect
        """
        if alpha <= 0.01:
            return frame

        h, w = frame.shape[:2]

        # Update pulse animation
        self._speaking_pulse = (self._speaking_pulse + 0.15) % (2 * math.pi)
        pulse = 0.5 + 0.5 * math.sin(self._speaking_pulse)

        # Detect stereo mode (aspect ratio > 2.0 means side-by-side)
        aspect = w / h
        is_stereo = aspect > 2.0

        if is_stereo:
            # Stereo mode: draw on both left and right halves
            half_w = w // 2

            # Left eye - head_x is relative to left half
            left_center_x = int(head_x * half_w)
            frame = self._draw_speaking_indicator_single(
                frame, left_center_x, head_y, is_speaking, alpha, pulse
            )

            # Right eye - same relative position in right half
            right_center_x = half_w + int(head_x * half_w)
            frame = self._draw_speaking_indicator_single(
                frame, right_center_x, head_y, is_speaking, alpha, pulse
            )
        else:
            # Mono mode
            center_x = int(head_x * w)
            frame = self._draw_speaking_indicator_single(
                frame, center_x, head_y, is_speaking, alpha, pulse
            )

        return frame

    def _draw_speaking_indicator_at_pos(self, frame: np.ndarray, ind_x: int, ind_y: int,
                                         is_speaking: bool, alpha: float = 1.0) -> np.ndarray:
        """Draw speaking indicator at a specific position (top-left corner).

        Args:
            frame: Video frame to draw on
            ind_x: X position of indicator (left edge)
            ind_y: Y position of indicator (top edge)
            is_speaking: Whether the other person is speaking
            alpha: Animation alpha for fade effect
        """
        if alpha <= 0.01:
            return frame

        h, w = frame.shape[:2]

        # Update pulse animation
        self._speaking_pulse = (self._speaking_pulse + 0.15) % (2 * math.pi)
        pulse = 0.5 + 0.5 * math.sin(self._speaking_pulse)

        indicator_width = self.indicator_width
        indicator_height = self.indicator_height
        radius = self.indicator_radius

        # Clamp to bounds
        ind_x = max(5, min(ind_x, w - indicator_width - 5))
        ind_y = max(5, min(ind_y, h - indicator_height - 5))

        # --- Glassmorphism background ---
        x1, y1 = max(0, ind_x - 2), max(0, ind_y - 2)
        x2, y2 = min(w, ind_x + indicator_width + 2), min(h, ind_y + indicator_height + 2)

        if x2 > x1 and y2 > y1:
            # Create rounded rect mask
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            mask_w, mask_h = x2 - x1, y2 - y1

            # Draw rounded rectangle in mask
            cv2.rectangle(mask, (radius, 0), (mask_w - radius, mask_h), 255, -1)
            cv2.rectangle(mask, (0, radius), (mask_w, mask_h - radius), 255, -1)
            cv2.circle(mask, (radius, radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius, radius), radius, 255, -1)
            cv2.circle(mask, (radius, mask_h - radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius, mask_h - radius), radius, 255, -1)

            # Get region and fast blur
            region = frame[y1:y2, x1:x2].copy()
            rh, rw = region.shape[:2]
            if rw > 8 and rh > 8:
                small = cv2.resize(region, (rw // 4, rh // 4), interpolation=cv2.INTER_AREA)
                small_blur = cv2.GaussianBlur(small, (5, 5), 0)
                blurred = cv2.resize(small_blur, (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                blurred = cv2.GaussianBlur(region, (7, 7), 0)

            # Dark tint blend
            blended = cv2.addWeighted(blurred, 0.25, np.full_like(blurred, (22, 22, 28), dtype=np.uint8), 0.75, 0)

            # Apply with mask
            mask_float = mask.astype(np.float32) * (0.88 * alpha / 255.0)
            mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
            result = blended.astype(np.float32) * mask_3ch + region.astype(np.float32) * (1.0 - mask_3ch)
            frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)

            # Subtle border
            border_color = (255, 255, 255)
            border_region = frame[y1:y2, x1:x2].copy()
            ox, oy = ind_x - x1, ind_y - y1

            cv2.line(border_region, (ox + radius, oy), (ox + indicator_width - radius, oy), border_color, 1)
            cv2.line(border_region, (ox + radius, oy + indicator_height), (ox + indicator_width - radius, oy + indicator_height), border_color, 1)
            cv2.line(border_region, (ox, oy + radius), (ox, oy + indicator_height - radius), border_color, 1)
            cv2.line(border_region, (ox + indicator_width, oy + radius), (ox + indicator_width, oy + indicator_height - radius), border_color, 1)

            cv2.ellipse(border_region, (ox + radius, oy + radius), (radius, radius), 180, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + indicator_width - radius, oy + radius), (radius, radius), 270, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + radius, oy + indicator_height - radius), (radius, radius), 90, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + indicator_width - radius, oy + indicator_height - radius), (radius, radius), 0, 0, 90, border_color, 1)

            frame[y1:y2, x1:x2] = cv2.addWeighted(border_region, 0.15 * alpha, frame[y1:y2, x1:x2], 1 - 0.15 * alpha, 0)

        # --- Draw content (dot + text) ---
        dot_size = max(8, int(10 * self._cached_scale))
        text_gap = max(8, int(10 * self._cached_scale))
        text = "SPEAKING" if is_speaking else "YOUR TURN"

        # Measure text using font metrics for proper vertical centering
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=self._font_indicator)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_y_offset = bbox[1]  # Top offset from baseline

        # Center content horizontally and vertically
        content_width = (dot_size * 2) + text_gap + text_w
        content_start_x = ind_x + (indicator_width - content_width) // 2
        indicator_center_y = ind_y + indicator_height // 2

        dot_center_x = content_start_x + dot_size
        dot_y = indicator_center_y
        text_x = content_start_x + (dot_size * 2) + text_gap

        # Draw pulsing dot
        if is_speaking:
            dot_color = (50, 205, 50)  # Green
            current_dot_size = int(dot_size * (0.85 + 0.15 * pulse))
        else:
            dot_color = (70, 130, 180)  # Steel blue
            current_dot_size = dot_size

        cv2.circle(frame, (dot_center_x, dot_y), current_dot_size, dot_color, -1, cv2.LINE_AA)

        # Draw text with PIL - use indicator bounds directly for proper centering
        tx1, ty1 = max(0, ind_x), max(0, ind_y)
        tx2, ty2 = min(w, ind_x + indicator_width), min(h, ind_y + indicator_height)

        if tx2 > tx1 and ty2 > ty1:
            text_region = frame[ty1:ty2, tx1:tx2].copy()
            pil_image = Image.fromarray(cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Calculate text position relative to region
            local_text_x = text_x - tx1
            # Center vertically: position text so its visual center aligns with indicator center
            region_center_y = (ty2 - ty1) // 2
            local_text_y = region_center_y - text_h // 2 - text_y_offset

            text_color = (255, 255, 255, int(230 * alpha))
            draw.text((local_text_x, local_text_y), text, font=self._font_indicator, fill=text_color)

            frame[ty1:ty2, tx1:tx2] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return frame

    def _update_toasts(self, state: dict) -> None:
        """Update toast notifications from state."""
        now = time.time()

        # Check for new social cue
        social_cue = state.get('social_cue')
        social_cue_icon = state.get('social_cue_icon')
        social_cue_timestamp = state.get('social_cue_timestamp', 0)

        # Only show toast if:
        # 1. There's a social cue
        # 2. It's newer than what we've seen
        # 3. It's recent (within last 5 seconds) - ignores stale state from previous runs
        is_new = social_cue_timestamp > self._last_social_cue_timestamp
        is_recent = (now - social_cue_timestamp) < 5.0

        if social_cue and is_new and is_recent:
            self._last_social_cue_timestamp = social_cue_timestamp
            self._active_toasts.append(ToastNotification(
                icon=social_cue_icon or "!",
                message=social_cue,
                timestamp=now
            ))

        # Remove expired toasts
        self._active_toasts = [t for t in self._active_toasts if now - t.timestamp < t.duration]

    def _draw_toast_single(self, frame: np.ndarray, toast, center_x: int,
                            base_y: int, toast_alpha: float) -> np.ndarray:
        """Draw a glassmorphism toast notification at specified position."""
        h, w = frame.shape[:2]

        # Dynamic width with LARGE padding to guarantee no overflow
        toast_height = self.toast_height
        dot_radius = max(10, int(12 * self._cached_scale))
        visual_margin = max(50, int(60 * self._cached_scale))
        gap_after_dot = max(40, int(50 * self._cached_scale))

        # Calculate max text width - very conservative
        max_width = int(600 * self._cached_scale)
        safety_buffer = max(60, int(80 * self._cached_scale))
        max_text_width = max_width - visual_margin - (dot_radius * 2) - gap_after_dot - visual_margin - safety_buffer

        # Truncate message to fit available width
        message = toast.message
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        # Measure and truncate if needed
        bbox = temp_draw.textbbox((0, 0), message, font=self._font_toast)
        text_w = bbox[2] - bbox[0]

        while text_w > max_text_width and len(message) > 3:
            message = message[:-4] + "..."
            bbox = temp_draw.textbbox((0, 0), message, font=self._font_toast)
            text_w = bbox[2] - bbox[0]

        text_h = bbox[3] - bbox[1]

        # Total width: left_margin + dot_diameter + gap + text + right_margin
        toast_width = visual_margin + (dot_radius * 2) + gap_after_dot + text_w + visual_margin

        # Minimum width (scaled)
        min_width = int(100 * self._cached_scale)
        toast_width = max(min_width, toast_width)
        radius = self.toast_radius

        toast_x = center_x - toast_width // 2
        toast_y = base_y

        # Clamp to bounds (scaled margin)
        bound_margin = int(10 * self._cached_scale)
        toast_x = max(bound_margin, min(toast_x, w - toast_width - bound_margin))

        # --- Glassmorphism background ---
        x1, y1 = max(0, toast_x - 2), max(0, toast_y - 2)
        x2, y2 = min(w, toast_x + toast_width + 2), min(h, toast_y + toast_height + 2)

        if x2 > x1 and y2 > y1:
            # Create rounded rect mask
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            mask_w, mask_h = x2 - x1, y2 - y1

            cv2.rectangle(mask, (radius, 0), (mask_w - radius, mask_h), 255, -1)
            cv2.rectangle(mask, (0, radius), (mask_w, mask_h - radius), 255, -1)
            cv2.circle(mask, (radius, radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius, radius), radius, 255, -1)
            cv2.circle(mask, (radius, mask_h - radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius, mask_h - radius), radius, 255, -1)

            # Get region and fast blur (downscale → blur → upscale)
            region = frame[y1:y2, x1:x2].copy()
            rh, rw = region.shape[:2]
            if rw > 8 and rh > 8:
                small = cv2.resize(region, (rw // 4, rh // 4), interpolation=cv2.INTER_AREA)
                small_blur = cv2.GaussianBlur(small, (5, 5), 0)
                blurred = cv2.resize(small_blur, (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                blurred = cv2.GaussianBlur(region, (7, 7), 0)

            # Dark tint blend
            blended = cv2.addWeighted(blurred, 0.25, np.full_like(blurred, (22, 22, 28), dtype=np.uint8), 0.75, 0)

            # Apply with mask
            mask_float = mask.astype(np.float32) * (0.88 * toast_alpha / 255.0)
            mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
            result = blended.astype(np.float32) * mask_3ch + region.astype(np.float32) * (1.0 - mask_3ch)
            frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)

            # Subtle border - only copy toast region, not entire frame
            border_color = (255, 255, 255)
            border_region = frame[y1:y2, x1:x2].copy()

            # Offset for drawing within the region
            ox, oy = toast_x - x1, toast_y - y1

            cv2.line(border_region, (ox + radius, oy), (ox + toast_width - radius, oy), border_color, 1)
            cv2.line(border_region, (ox + radius, oy + toast_height), (ox + toast_width - radius, oy + toast_height), border_color, 1)
            cv2.line(border_region, (ox, oy + radius), (ox, oy + toast_height - radius), border_color, 1)
            cv2.line(border_region, (ox + toast_width, oy + radius), (ox + toast_width, oy + toast_height - radius), border_color, 1)

            cv2.ellipse(border_region, (ox + radius, oy + radius), (radius, radius), 180, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + toast_width - radius, oy + radius), (radius, radius), 270, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + radius, oy + toast_height - radius), (radius, radius), 90, 0, 90, border_color, 1)
            cv2.ellipse(border_region, (ox + toast_width - radius, oy + toast_height - radius), (radius, radius), 0, 0, 90, border_color, 1)

            frame[y1:y2, x1:x2] = cv2.addWeighted(border_region, 0.15 * toast_alpha, frame[y1:y2, x1:x2], 1 - 0.15 * toast_alpha, 0)

        # --- Accent indicator (small glowing dot) ---
        # Position dot so its LEFT EDGE is at visual_margin from toast edge
        dot_x = toast_x + visual_margin + dot_radius
        dot_y = toast_y + toast_height // 2
        glow_radius = int(dot_radius * 2)

        # Amber/gold glow - only copy glow region, not entire frame
        gr = glow_radius + 2
        gx1, gy1 = max(0, dot_x - gr), max(0, dot_y - gr)
        gx2, gy2 = min(w, dot_x + gr), min(h, dot_y + gr)
        if gx2 > gx1 and gy2 > gy1:
            glow_region = frame[gy1:gy2, gx1:gx2].copy()
            cv2.circle(glow_region, (dot_x - gx1, dot_y - gy1), glow_radius, (80, 180, 240), -1)
            frame[gy1:gy2, gx1:gx2] = cv2.addWeighted(glow_region, 0.25 * toast_alpha, frame[gy1:gy2, gx1:gx2], 1 - 0.25 * toast_alpha, 0)

        # Solid dot
        cv2.circle(frame, (dot_x, dot_y), dot_radius, (100, 200, 255), -1)  # BGR - amber
        cv2.circle(frame, (dot_x, dot_y), dot_radius, (150, 220, 255), 1)   # Highlight

        # --- Text with PIL - only convert toast region, not entire frame ---
        # Calculate text position - account for font bearing (bbox offset)
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), message, font=self._font_toast)
        text_left_bearing = bbox[0]

        # Text draw position: dot right edge + gap - left bearing
        dot_right_edge = dot_x + dot_radius
        text_x = dot_right_edge + gap_after_dot - text_left_bearing
        text_y = dot_y - text_h // 2 - bbox[1]

        # Region for PIL text rendering (toast bounds with padding)
        pad = 5
        tx1, ty1 = max(0, toast_x - pad), max(0, toast_y - pad)
        tx2, ty2 = min(w, toast_x + toast_width + pad), min(h, toast_y + toast_height + pad)

        if tx2 > tx1 and ty2 > ty1:
            text_region = frame[ty1:ty2, tx1:tx2]
            text_region_rgb = cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(text_region_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Adjust text position for region offset
            local_text_x = text_x - tx1
            local_text_y = text_y - ty1

            text_color = (255, 255, 255, int(240 * toast_alpha))
            draw.text((local_text_x, local_text_y), message, font=self._font_toast, fill=text_color)

            frame[ty1:ty2, tx1:tx2] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return frame

    def _draw_toasts(self, frame: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Draw toast notifications at the bottom of the screen.

        Handles stereo mode by drawing on both halves.
        """
        if not self._active_toasts or alpha <= 0.01:
            return frame

        h, w = frame.shape[:2]
        now = time.time()

        # Detect stereo mode
        aspect = w / h
        is_stereo = aspect > 2.0

        # Use Quest-style positioning if stereo OR if force_quest_mode is enabled
        use_quest_positioning = is_stereo or self.force_quest_mode

        # Draw each toast (only most recent one to avoid clutter)
        toast = self._active_toasts[-1]  # Most recent toast

        # Calculate fade based on age
        age = now - toast.timestamp
        if age < 0.3:
            toast_alpha = age / 0.3  # Fade in
        elif age > toast.duration - 0.5:
            toast_alpha = (toast.duration - age) / 0.5  # Fade out
        else:
            toast_alpha = 1.0

        toast_alpha *= alpha

        # Position toast - higher for Quest to avoid being cut off
        if use_quest_positioning:
            base_y = h - int(120 * self._cached_scale)  # Higher for Quest
        else:
            base_y = h - int(55 * self._cached_scale)  # Position near bottom (scaled)

        if is_stereo:
            half_w = w // 2
            # Draw on left eye
            frame = self._draw_toast_single(frame, toast, half_w // 2, base_y, toast_alpha)
            # Draw on right eye
            frame = self._draw_toast_single(frame, toast, half_w + half_w // 2, base_y, toast_alpha)
        else:
            frame = self._draw_toast_single(frame, toast, w // 2, base_y, toast_alpha)

        return frame

    def render(self, frame: np.ndarray, head_x: float = 0.5, head_y: float = 0.5,
               person_tracked: bool = False) -> np.ndarray:
        """Render overlay panels with animation."""
        # Profiling
        profile_enabled = getattr(self, '_profile_enabled', True)
        if profile_enabled:
            t_start = time.time()
            timings = {}

        h, w = frame.shape[:2]

        # Update scale based on frame resolution
        if profile_enabled: t0 = time.time()
        self._update_scale(h)
        if profile_enabled: timings['scale'] = (time.time() - t0) * 1000

        # Update animation
        if profile_enabled: t0 = time.time()
        anim_progress = self._update_animation(person_tracked)
        if profile_enabled: timings['anim'] = (time.time() - t0) * 1000

        # Load state (always load for toasts, even when hidden)
        if profile_enabled: t0 = time.time()
        state = self._load_state()
        if profile_enabled: timings['state'] = (time.time() - t0) * 1000

        # Update and draw toast notifications (always visible)
        if profile_enabled: t0 = time.time()
        self._update_toasts(state)
        frame = self._draw_toasts(frame, alpha=1.0)
        if profile_enabled: timings['toasts'] = (time.time() - t0) * 1000

        # Skip rest of rendering if fully hidden
        if anim_progress <= 0.01:
            if profile_enabled:
                self._log_profile(timings, time.time() - t_start)
            return frame

        # Get state values
        convo_summary = state.get("convo_state_summary") or state.get("conversation_summary") or "Listening..."
        question = state.get("question", "")
        utterance = state.get("recent_utterance", "")
        # Use emotion_display if available, fallback to emotion (no emojis)
        raw_emotion = state.get("emotion_display") or state.get("emotion") or state.get("emotion_label") or "Neutral"
        is_speaking = state.get("is_other_speaking", False)

        # Get recent utterances list for stacked captions (baseball-style rolling captions)
        utterances = state.get("recent_utterances", [])
        if not utterances and utterance:
            utterances = [utterance]  # Fallback to single utterance

        # Map emotions and get corresponding emoji
        emotion, emotion_emoji = self._map_emotion(raw_emotion)

        # Detect stereo mode
        aspect = w / h
        is_stereo = aspect > 2.0

        # Use Quest-style positioning if stereo OR if force_quest_mode is enabled
        # This allows mono videos meant for Quest viewing to have proper positioning
        use_quest_positioning = is_stereo or self.force_quest_mode

        # Calculate panel dimensions first (needed for speaking indicator positioning)
        line_height_text = int(24 * self._cached_scale)
        line_height_label = int(18 * self._cached_scale)
        section_spacing = int(16 * self._cached_scale)

        # Calculate context panel height dynamically
        context_sections = 3  # conversation, question, just said
        context_height = (self.padding_y * 2) + (context_sections * (line_height_label + line_height_text * 2)) + (section_spacing * 2)
        emotion_height = self.padding_y * 2 + line_height_label + int(32 * self._cached_scale)

        total_height = context_height + self.panel_gap + emotion_height

        # Animated position (slide in from left) - scaled
        # Use larger offset for Quest/stereo to move panels closer to center
        if use_quest_positioning:
            base_x = int(200 * self._cached_scale)  # Further from center for Quest
        else:
            base_x = int(28 * self._cached_scale)
        slide_offset = int((1 - anim_progress) * -70 * self._cached_scale)  # Slide from left

        # For stereo mode, render panels on both halves
        if is_stereo:
            half_w = w // 2
            # Panel width 25% smaller for Quest, and move over by same amount
            panel_width = int(min(self.panel_width, half_w - 60) * 0.75)
            base_x += int(self.panel_width * 0.25)  # Move over by 25% of original width

            # Draw on left eye
            if profile_enabled: t0 = time.time()
            frame = self._draw_panels_single(
                frame, base_x + slide_offset, h, panel_width,
                context_height, emotion_height, total_height,
                anim_progress, convo_summary, question, utterances, emotion, emotion_emoji,
                timings if profile_enabled else None
            )
            if profile_enabled: timings['panel_L'] = (time.time() - t0) * 1000

            # Draw on right eye (offset by half width)
            if profile_enabled: t0 = time.time()
            frame = self._draw_panels_single(
                frame, half_w + base_x + slide_offset, h, panel_width,
                context_height, emotion_height, total_height,
                anim_progress, convo_summary, question, utterances, emotion, emotion_emoji,
                timings if profile_enabled else None
            )
            if profile_enabled: timings['panel_R'] = (time.time() - t0) * 1000
        else:
            # Mono mode - use Quest-style panel width if force_quest_mode is enabled
            if profile_enabled: t0 = time.time()
            if use_quest_positioning:
                # Use 75% panel width like stereo mode for Quest-friendly positioning
                mono_panel_width = int(self.panel_width * 0.75)
                mono_base_x = base_x  # base_x already positioned for Quest, no additional offset
            else:
                mono_panel_width = self.panel_width
                mono_base_x = base_x
            frame = self._draw_panels_single(
                frame, mono_base_x + slide_offset, h, mono_panel_width,
                context_height, emotion_height, total_height,
                anim_progress, convo_summary, question, utterances, emotion, emotion_emoji,
                timings if profile_enabled else None
            )
            if profile_enabled: timings['panel'] = (time.time() - t0) * 1000

        # Draw speaking indicator on the RIGHT side, vertically centered with panels
        if profile_enabled: t0 = time.time()
        # Panels are vertically centered in frame (same calc as _draw_panels_single)
        panel_base_y = (h - total_height) // 2
        # Calculate vertical center of panels, then position indicator so its center aligns
        panel_vertical_center = panel_base_y + total_height // 2
        indicator_y = panel_vertical_center - self.indicator_height // 2

        if is_stereo:
            half_w = w // 2
            eye_center = half_w // 2  # Center of each eye's view

            # Panel position and width (with animation offset)
            panel_left_x = base_x + slide_offset
            # Use the scaled panel width for stereo mode
            stereo_panel_width = int(min(self.panel_width, half_w - 60) * 0.75)

            # Calculate panel center position
            panel_center_x = panel_left_x + stereo_panel_width // 2

            # Distance from panel center to eye center
            dist_from_center = eye_center - panel_center_x

            # Mirror: indicator's CENTER should be same distance to the RIGHT of center
            indicator_center_x_left = eye_center + dist_from_center
            indicator_x_left = indicator_center_x_left - self.indicator_width // 2

            indicator_center_x_right = half_w + eye_center + dist_from_center
            indicator_x_right = indicator_center_x_right - self.indicator_width // 2

            frame = self._draw_speaking_indicator_at_pos(
                frame, indicator_x_left, indicator_y, is_speaking, anim_progress
            )
            frame = self._draw_speaking_indicator_at_pos(
                frame, indicator_x_right, indicator_y, is_speaking, anim_progress
            )
        else:
            # Mono mode: mirror around frame center
            frame_center = w // 2

            # Use same panel positioning as rendering (Quest mode uses adjusted values)
            if use_quest_positioning:
                mono_panel_width = int(self.panel_width * 0.75)
                mono_base_x = base_x  # base_x already positioned for Quest, no additional offset
            else:
                mono_panel_width = self.panel_width
                mono_base_x = base_x

            panel_left_x = mono_base_x + slide_offset

            # Calculate panel center position
            panel_center_x = panel_left_x + mono_panel_width // 2

            # Distance from panel center to frame center
            dist_from_center = frame_center - panel_center_x

            # Mirror: indicator's CENTER should be same distance to the RIGHT of center
            indicator_center_x = frame_center + dist_from_center
            indicator_x = indicator_center_x - self.indicator_width // 2

            frame = self._draw_speaking_indicator_at_pos(
                frame, indicator_x, indicator_y, is_speaking, anim_progress
            )
        if profile_enabled: timings['speaking_ind'] = (time.time() - t0) * 1000

        if profile_enabled:
            self._log_profile(timings, time.time() - t_start)

        return frame

    def _log_profile(self, timings: dict, total_ms: float) -> None:
        """Log profiling data periodically."""
        if not hasattr(self, '_profile_history'):
            self._profile_history = []
            self._profile_last_log = time.time()

        self._profile_history.append((timings, total_ms * 1000))

        # Log every 2 seconds
        if time.time() - self._profile_last_log >= 2.0 and self._profile_history:
            # Average all timings
            avg_timings = {}
            avg_total = 0
            for t, total in self._profile_history:
                for k, v in t.items():
                    avg_timings[k] = avg_timings.get(k, 0) + v
                avg_total += total

            n = len(self._profile_history)
            for k in avg_timings:
                avg_timings[k] /= n
            avg_total /= n

            # Format output
            parts = [f"{k}:{v:.1f}ms" for k, v in sorted(avg_timings.items(), key=lambda x: -x[1])]
            print(f"[OVERLAY] {' | '.join(parts)} | TOTAL:{avg_total:.1f}ms")

            self._profile_history.clear()
            self._profile_last_log = time.time()

    def _draw_panels_single(self, frame: np.ndarray, panel_x: int, frame_h: int,
                            panel_width: int, context_height: int, emotion_height: int,
                            total_height: int, anim_progress: float,
                            convo_summary: str, question: str, utterances: List[str],
                            emotion: str, emotion_emoji: str,
                            _timings: dict = None) -> np.ndarray:
        """Draw the info panels at a specific x position.

        Args:
            utterances: List of recent utterances for stacked/rolling captions (baseball-style).
                       Most recent utterance should be last in the list (displayed at bottom).
        """
        _t = _timings is not None

        line_height_text = int(24 * self._cached_scale)
        line_height_label = int(18 * self._cached_scale)
        section_spacing = int(16 * self._cached_scale)

        base_y = (frame_h - total_height) // 2

        # Draw glassmorphism panels
        context_y = base_y
        frame = self._draw_glassmorphism_panel(frame, panel_x, context_y,
                                                panel_width, context_height, anim_progress, _timings)

        # Emotion panel: split into two halves with gap equal to panel_gap
        emotion_y = context_y + context_height + self.panel_gap
        emotion_half_width = (panel_width - self.panel_gap) // 2

        # Left half: emotion text
        frame = self._draw_glassmorphism_panel(frame, panel_x, emotion_y,
                                                emotion_half_width, emotion_height, anim_progress, _timings)
        # Right half: emoji
        emoji_x = panel_x + emotion_half_width + self.panel_gap
        frame = self._draw_glassmorphism_panel(frame, emoji_x, emotion_y,
                                                emotion_half_width, emotion_height, anim_progress, _timings)

        # Render text with PIL - only convert panel region, not entire frame
        # Calculate panel region bounds (all panels including emotion)
        h, w = frame.shape[:2]
        pad = 5
        rx1, ry1 = max(0, panel_x - pad), max(0, base_y - pad)
        rx2, ry2 = min(w, panel_x + panel_width + pad), min(h, base_y + total_height + pad)

        if rx2 <= rx1 or ry2 <= ry1:
            return frame

        if _t: t0 = time.time()
        panel_region = frame[ry1:ry2, rx1:rx2]
        panel_region_rgb = cv2.cvtColor(panel_region, cv2.COLOR_BGR2RGB)
        if _t: _timings['pil_cvt2rgb'] = _timings.get('pil_cvt2rgb', 0) + (time.time() - t0) * 1000

        if _t: t0 = time.time()
        pil_image = Image.fromarray(panel_region_rgb)
        if _t: _timings['pil_fromarray'] = _timings.get('pil_fromarray', 0) + (time.time() - t0) * 1000

        draw = ImageDraw.Draw(pil_image)

        # Text alpha based on animation
        text_alpha = int(255 * anim_progress)
        white_a = (255, 255, 255, text_alpha)
        label_a = (160, 160, 160, text_alpha)
        dim_a = (140, 140, 140, text_alpha)

        # Adjust coordinates for region offset
        local_panel_x = panel_x - rx1
        local_context_y = context_y - ry1
        local_emotion_y = emotion_y - ry1
        local_emoji_x = emoji_x - rx1

        text_x = local_panel_x + self.padding_x
        max_text_width = panel_width - (self.padding_x * 2)

        # === CONTEXT PANEL ===
        y = local_context_y + self.padding_y

        # CONVERSATION
        draw.text((text_x, y), "CONVERSATION", font=self._font_label, fill=label_a)
        y += line_height_label

        summary_lines = self._wrap_text(draw, convo_summary, self._font_text, max_text_width)
        for line in summary_lines[:2]:
            draw.text((text_x, y), line, font=self._font_text, fill=white_a)
            y += line_height_text
        y += section_spacing

        # QUESTION
        draw.text((text_x, y), "QUESTION", font=self._font_label, fill=label_a)
        y += line_height_label

        q_text = question if question else "No question detected"
        q_fill = white_a if question else dim_a
        q_lines = self._wrap_text(draw, q_text, self._font_text, max_text_width)
        for line in q_lines[:2]:
            draw.text((text_x, y), line, font=self._font_text, fill=q_fill)
            y += line_height_text
        y += section_spacing

        # JUST SAID - stacked/rolling captions like baseball game captions
        draw.text((text_x, y), "JUST SAID", font=self._font_label, fill=label_a)
        y += line_height_label

        # Use smaller line height for utterances to fit more lines
        utt_line_height = int(20 * self._cached_scale)

        if not utterances:
            # No utterances - show "Listening..." instead of "..."
            draw.text((text_x, y), "Listening...", font=self._font_text, fill=dim_a)
            y += utt_line_height
        else:
            # Display stacked utterances - older ones dimmer, most recent at bottom
            # Limit to 2 utterances max, 1 line each for compact display
            max_utterances = 2
            display_utterances = utterances[-2:]  # Take last N (most recent)

            for i, utt in enumerate(display_utterances):
                # Calculate opacity: older utterances are dimmer
                # First utterance (oldest) is dimmest, last (most recent) is brightest
                is_most_recent = (i == len(display_utterances) - 1)

                if is_most_recent:
                    # Most recent utterance - full brightness
                    utt_fill = white_a
                else:
                    # Older utterances - progressively dimmer
                    # Calculate dimness based on position (0 = oldest, len-1 = newest)
                    dim_factor = 0.5 + (0.3 * i / max(1, len(display_utterances) - 1))
                    dim_value = int(180 * dim_factor)
                    utt_fill = (dim_value, dim_value, dim_value, text_alpha)

                # Format utterance with quotes
                utt_text = f'"{utt}"' if utt else ""

                # Wrap text and allow only 1 line per utterance to prevent overflow
                utt_lines = self._wrap_text(draw, utt_text, self._font_text, max_text_width)
                for line in utt_lines[:1]:  # Only 1 line per utterance to fit in panel
                    draw.text((text_x, y), line, font=self._font_text, fill=utt_fill)
                    y += utt_line_height

        # === EMOTION PANEL (split into two halves) ===
        y = local_emotion_y + self.padding_y

        # Left half: emotion label and text
        draw.text((text_x, y), "EMOTION", font=self._font_label, fill=label_a)
        emotion_text_y = y + line_height_label + int(4 * self._cached_scale)

        # Dynamic font sizing for emotion text - shrink if too wide
        emotion_max_width = emotion_half_width - (self.padding_x * 2)
        emotion_font = self._font_emotion
        emotion_font_size = self.emotion_size

        # Measure and shrink font if needed
        bbox = draw.textbbox((0, 0), emotion, font=emotion_font)
        text_width = bbox[2] - bbox[0]

        while text_width > emotion_max_width and emotion_font_size > 12:
            emotion_font_size -= 2
            emotion_font = self._load_font(emotion_font_size, bold=True)
            bbox = draw.textbbox((0, 0), emotion, font=emotion_font)
            text_width = bbox[2] - bbox[0]

        draw.text((text_x, emotion_text_y), emotion, font=emotion_font, fill=white_a)

        # Right half: emoji centered
        emoji_center_x = local_emoji_x + emotion_half_width // 2
        emoji_center_y = local_emotion_y + emotion_height // 2

        # Measure emoji to center it
        try:
            emoji_bbox = draw.textbbox((0, 0), emotion_emoji, font=self._font_emoji)
            emoji_w = emoji_bbox[2] - emoji_bbox[0]
            emoji_h = emoji_bbox[3] - emoji_bbox[1]
            emoji_draw_x = emoji_center_x - emoji_w // 2 - emoji_bbox[0]
            emoji_draw_y = emoji_center_y - emoji_h // 2 - emoji_bbox[1]
            draw.text((emoji_draw_x, emoji_draw_y), emotion_emoji, font=self._font_emoji, fill=white_a)
        except:
            # Fallback if emoji rendering fails
            draw.text((local_emoji_x + self.padding_x, emotion_text_y), emotion_emoji, font=self._font_emotion, fill=white_a)

        # Convert back to BGR - only the panel region
        if _t: t0 = time.time()
        arr = np.array(pil_image)
        if _t: _timings['pil_toarray'] = _timings.get('pil_toarray', 0) + (time.time() - t0) * 1000

        if _t: t0 = time.time()
        frame[ry1:ry2, rx1:rx2] = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if _t: _timings['pil_cvt2bgr'] = _timings.get('pil_cvt2bgr', 0) + (time.time() - t0) * 1000

        return frame


# Global renderer
_renderer: Optional[OverlayRenderer] = None


def get_renderer() -> OverlayRenderer:
    """Get or create the overlay renderer."""
    global _renderer
    if _renderer is None:
        _renderer = OverlayRenderer()
    return _renderer


def render_overlay(frame: np.ndarray, head_x: float = 0.5, head_y: float = 0.5,
                   person_tracked: bool = False) -> np.ndarray:
    """Render overlay panels onto frame."""
    renderer = get_renderer()
    return renderer.render(frame, head_x, head_y, person_tracked)


def start_overlay():
    """No-op for backwards compatibility."""
    pass


def stop_overlay():
    """No-op for backwards compatibility."""
    pass


def update_head_position(head_y: float):
    """No-op for backwards compatibility."""
    pass
