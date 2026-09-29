"""
streamlit_app.py

Streamlit UI for the NavSense AI prototype.

This file is ONLY responsible for how things look and are laid out on screen.
All actual AI behavior is unchanged and still comes from:
    - detection.py  (YOLOv8 object detection)
    - direction.py  (LEFT / CENTER / RIGHT calculation)

The page is organized into three sections, switched by a horizontal nav bar:
    - Live Detection : the camera + real-time detection (default landing page)
    - Guidance        : a plain-language summary of the most recent detections
    - About           : a short explanation of what NavSense AI is
"""

import textwrap
import time
import urllib.parse

import cv2
import streamlit as st

from detection import ObjectDetector
from direction import get_direction


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NavSense AI",
    page_icon="◎",
    layout="wide",
)

NAV_ITEMS = ["Live Detection", "Guidance", "About"]


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLOR_BG = "#F7F1E3"            # Warm ivory page background
COLOR_SURFACE = "#FFFDF7"       # Clean paper-like panels
COLOR_SURFACE_ALT = "#EFE5C9"   # Pressed-petal secondary surface
COLOR_BORDER = "#B9AA78"        # Soft botanical border
COLOR_BORDER_FOCUS = "#3F6B45"  # Leaf-green focus color
COLOR_TEXT = "#2E2A22"          # Warm charcoal text
COLOR_TEXT_MUTED = "#706757"    # Muted bark text
COLOR_ACCENT = "#3F6B45"        # Leaf green primary accent
COLOR_ACCENT_SOFT = "rgba(63, 107, 69, 0.12)" # Soft leaf tint
COLOR_AMBER = "#C87820"         # Marigold directional accent
COLOR_SUCCESS = "#568044"       # Fresh leaf active state
COLOR_ERROR = "#A7442E"         # Earthy terracotta error state

# BGR tuple for OpenCV camera bounding box overlay (high-visibility marigold)
BOX_COLOR_BGR = (32, 120, 200)


# ---------------------------------------------------------------------------
# Small helpers for rendering custom HTML correctly
# ---------------------------------------------------------------------------
def _clean_html(html_str: str) -> str:
    """Prepares a hand-written HTML string for st.markdown()."""
    dedented = textwrap.dedent(html_str).strip()
    return "\n".join(line for line in dedented.splitlines() if line.strip())


def render_html(html_str: str):
    """Renders a block of custom HTML."""
    st.markdown(_clean_html(html_str), unsafe_allow_html=True)


def render_html_into(placeholder, html_str: str):
    """Same as render_html(), but writes into an existing st.empty() placeholder."""
    placeholder.markdown(_clean_html(html_str), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
def inject_css():
    render_html(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
        html, body, [class*="css"] {{
            font-family: 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: {COLOR_TEXT};
        }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {COLOR_BG};
        }}

        /* Hide Streamlit's default chrome */
        #MainMenu, footer {{ visibility: hidden; }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            padding-left: 4.5rem;
            max-width: 1180px;
        }}

        /* Style Streamlit Native Bordered Containers with High Contrast Border */
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            background-color: {COLOR_SURFACE} !important;
            border: 2px solid {COLOR_BORDER} !important;
            border-radius: 12px !important;
            box-shadow: 0 14px 34px rgba(63, 79, 48, 0.10);
        }}

        /* Distinct font for "Camera settings" expander */
        [data-testid="stExpander"] summary p {{
            font-family: 'Manrope', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: {COLOR_ACCENT} !important;
        }}

        /* Widget label visibility */
        label, [data-testid="stWidgetLabel"] p {{
            color: {COLOR_TEXT} !important;
            font-weight: 500 !important;
        }}

        /* ---------- Brand ---------- */
        .ns-brand {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }}
        .ns-logo-mark {{
            width: 36px;
            height: 36px;
            flex-shrink: 0;
            border-radius: 10px;
            background: {COLOR_ACCENT_SOFT};
            color: {COLOR_ACCENT};
            border: 1px solid {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .ns-brand-text h1 {{
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0;
            color: {COLOR_TEXT};
            letter-spacing: 0;
        }}
        .ns-brand-text p {{
            font-size: 0.8rem;
            margin: 0;
            color: {COLOR_TEXT_MUTED};
        }}

        /* ---------- Nav bar ---------- */
        .ns-navbar-divider {{
            border-bottom: 1px solid {COLOR_BORDER};
            margin: 1.1rem 0 1.8rem 0;
        }}
        .ns-nav-active {{
            text-align: center;
            padding: 0.55rem 0.4rem 0.5rem 0.4rem;
            font-weight: 700;
            font-size: 0.95rem;
            color: {COLOR_ACCENT};
            border-bottom: 2px solid {COLOR_ACCENT};
        }}
        div[data-testid="stHorizontalBlock"] .stButton>button {{
            width: 100%;
            background: transparent;
            border: 1px solid transparent;
            box-shadow: none;
            border-radius: 6px;
            color: {COLOR_TEXT_MUTED};
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.55rem 0.4rem;
            transition: all 0.2s ease;
        }}
        div[data-testid="stHorizontalBlock"] .stButton>button:hover {{
            color: {COLOR_ACCENT};
            background: {COLOR_ACCENT_SOFT};
            border-color: {COLOR_BORDER};
        }}

        /* ---------- Status pill ---------- */
        .ns-status-wrap {{
            display: flex;
            justify-content: flex-end;
        }}
        .ns-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            border: 1.5px solid {COLOR_BORDER};
            background: {COLOR_SURFACE};
            font-size: 0.83rem;
            font-weight: 500;
            color: {COLOR_TEXT};
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }}
        .ns-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            flex-shrink: 0;
        }}

        /* ---------- Section labels / page headers ---------- */
        .ns-section-label {{
            font-size: 0.98rem;
            font-weight: 600;
            color: {COLOR_TEXT};
            margin-bottom: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .ns-page-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {COLOR_TEXT};
            margin: 0 0 0.3rem 0;
        }}
        .ns-page-subtitle {{
            font-size: 0.95rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0 0 1.4rem 0;
        }}

        /* ---------- Empty states ---------- */
        .ns-empty {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.9rem;
            padding: 4.5rem 1.5rem;
            text-align: center;
        }}
        .ns-empty-icon {{
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: {COLOR_ACCENT_SOFT};
            border: 1px solid {COLOR_BORDER};
            color: {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .ns-empty-title {{
            font-size: 1.05rem;
            font-weight: 600;
            color: {COLOR_TEXT};
            margin: 0;
        }}
        .ns-empty-subtitle {{
            font-size: 0.88rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0;
            max-width: 26em;
        }}

        /* ---------- Detected object list ---------- */
        .ns-object-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.75rem 0.5rem;
            border-bottom: 1px solid {COLOR_SURFACE_ALT};
            border-radius: 6px;
        }}
        .ns-object-row:last-child {{
            border-bottom: none;
        }}
        .ns-object-name {{
            font-size: 0.95rem;
            font-weight: 600;
            color: {COLOR_TEXT};
        }}
        .ns-object-confidence {{
            font-size: 0.8rem;
            color: {COLOR_TEXT_MUTED};
            margin-top: 0.15rem;
        }}
        .ns-direction-badge {{
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
            padding: 0.3rem 0.7rem;
            border-radius: 8px;
            letter-spacing: 0.02em;
        }}
        .ns-direction-left   {{ background: rgba(99, 61, 80, 0.12); color: #633D50; border: 1px solid #8A6575; }}
        .ns-direction-center {{ background: rgba(86, 128, 68, 0.14); color: {COLOR_SUCCESS}; border: 1px solid {COLOR_SUCCESS}; }}
        .ns-direction-right  {{ background: rgba(200, 120, 32, 0.14); color: {COLOR_AMBER}; border: 1px solid {COLOR_AMBER}; }}

        /* ---------- Guidance page ---------- */
        .ns-guidance-panel {{
            display: flex;
            gap: 1.2rem;
            align-items: flex-start;
            background: {COLOR_SURFACE};
            border: 2px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        .ns-guidance-icon {{
            width: 42px;
            height: 42px;
            flex-shrink: 0;
            border-radius: 12px;
            background: {COLOR_ACCENT_SOFT};
            border: 1px solid {COLOR_ACCENT};
            color: {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .ns-guidance-text {{
            font-size: 1.1rem;
            font-weight: 500;
            line-height: 1.6;
            color: {COLOR_TEXT};
            margin: 0;
        }}
        .ns-guidance-meta {{
            font-size: 0.82rem;
            color: {COLOR_TEXT_MUTED};
            margin-top: 0.8rem;
        }}

        /* ---------- About page ---------- */
        .ns-about-panel {{
            background: {COLOR_SURFACE};
            border: 2px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 1.8rem 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        .ns-about-panel p {{
            font-size: 0.98rem;
            line-height: 1.7;
            color: {COLOR_TEXT};
            margin: 0 0 1rem 0;
        }}
        .ns-about-panel p.ns-about-note {{
            color: {COLOR_TEXT_MUTED};
            font-size: 0.9rem;
        }}
        .ns-about-heading {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 1.6rem 0 0.8rem 0;
        }}
        .ns-about-heading svg {{
            color: {COLOR_ACCENT};
        }}
        .ns-about-heading h3 {{
            font-size: 1.05rem;
            font-weight: 600;
            color: {COLOR_TEXT};
            margin: 0;
        }}
        .ns-about-steps {{
            display: flex;
            gap: 1.2rem;
            margin: 1rem 0 1.6rem 0;
            flex-wrap: wrap;
        }}
        .ns-about-step {{
            flex: 1;
            min-width: 160px;
            background: {COLOR_SURFACE_ALT};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
        }}
        .ns-about-step-icon {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: {COLOR_ACCENT_SOFT};
            border: 1px solid {COLOR_ACCENT};
            color: {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.7rem;
        }}
        .ns-about-step h4 {{
            font-size: 0.92rem;
            font-weight: 600;
            color: {COLOR_TEXT};
            margin: 0 0 0.3rem 0;
        }}
        .ns-about-step p {{
            font-size: 0.85rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0;
            line-height: 1.5;
        }}

        /* ---------- Left brand rail ---------- */
        .ns-edge-rail {{
            position: fixed;
            top: 0;
            left: 0;
            width: 52px;
            height: 100vh;
            background-image: url("{RAIL_PATTERN_DATA_URI}"),
                linear-gradient(180deg, #3F6B45 0%, #315238 72%, #243D2A 100%);
            background-repeat: repeat-y, no-repeat;
            background-position: top center, center;
            background-size: 52px 120px, cover;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding-top: 1.6rem;
            gap: 1.1rem;
            z-index: 999;
            border-right: 1px solid {COLOR_BORDER};
        }}
        .ns-edge-mark {{
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: rgba(255, 253, 247, 0.12);
            color: {COLOR_ACCENT};
            border: 1px solid {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 5px 18px rgba(26, 48, 30, 0.28);
        }}
        .ns-edge-label {{
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.22em;
            color: #FFFDF7;
            white-space: nowrap;
            margin-bottom: 1rem;
        }}


        /* ---------- Botanical finish & responsive layout ---------- */
        .ns-brand-text h1, .ns-page-title, .ns-empty-title, .ns-about-heading h3 {{
            font-family: 'Fraunces', Georgia, serif;
        }}
        .ns-page-title {{
            font-size: clamp(1.9rem, 4vw, 3rem);
            line-height: 1.08;
        }}
        .ns-logo-mark, .ns-empty-icon, .ns-guidance-icon {{
            border-radius: 50%;
        }}
        .ns-logo-mark {{
            position: relative;
            box-shadow: 0 5px 16px rgba(63, 107, 69, 0.16);
        }}
        .ns-logo-mark::after {{
            content: '';
            position: absolute;
            width: 9px;
            height: 9px;
            right: -3px;
            top: -2px;
            border-radius: 50%;
            background: {COLOR_AMBER};
            border: 2px solid {COLOR_SURFACE};
        }}
        .ns-navbar-divider {{ border-color: rgba(98, 82, 52, 0.24); }}
        .ns-status-pill {{ box-shadow: 0 5px 18px rgba(63, 79, 48, 0.10); }}
        .ns-guidance-panel, .ns-about-panel {{
            border-width: 1px;
            box-shadow: 0 18px 48px rgba(63, 79, 48, 0.10);
        }}
        .ns-about-step {{ border-width: 1px; }}
        [data-testid="stAppViewContainer"] {{
            background-image:
                radial-gradient(circle at 92% 8%, rgba(200, 120, 32, 0.10) 0 4.5rem, transparent 4.6rem),
                radial-gradient(circle at 89% 5%, rgba(63, 107, 69, 0.08) 0 8rem, transparent 8.1rem);
        }}
        .stButton>button:focus, button:focus-visible, input:focus-visible {{
            outline: 3px solid rgba(63, 107, 69, 0.24) !important;
            outline-offset: 2px;
        }}
        @media (max-width: 760px) {{
            .block-container {{ padding: 1rem 1rem 2rem 1rem; }}
            .ns-edge-rail {{ display: none; }}
            .ns-page-title {{ margin-top: 0.5rem; }}
            .ns-status-wrap {{ justify-content: flex-start; margin-bottom: 0.8rem; }}
            .ns-guidance-panel, .ns-about-panel {{ padding: 1.2rem; }}
        }}

        /* Native Button styling tweak */
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid {COLOR_BORDER};
            background: {COLOR_SURFACE};
            color: {COLOR_TEXT};
        }}
    </style>
    """)


# SVG Icons
EYE_ICON_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7Z"/>
    <circle cx="12" cy="12" r="3"/>
</svg>
"""

CAMERA_ICON_SVG = """
<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 7h3l2-2h8l2 2h3v12H3z"/>
    <circle cx="12" cy="13" r="3.5"/>
</svg>
"""

LIST_ICON_SVG = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <line x1="8" y1="6" x2="20" y2="6"/><line x1="8" y1="12" x2="20" y2="12"/><line x1="8" y1="18" x2="20" y2="18"/>
    <circle cx="4" cy="6" r="1.2" fill="currentColor" stroke="none"/>
    <circle cx="4" cy="12" r="1.2" fill="currentColor" stroke="none"/>
    <circle cx="4" cy="18" r="1.2" fill="currentColor" stroke="none"/>
</svg>
"""

COMPASS_ICON_SVG = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/>
    <path d="M14.8 9.2 13 13l-3.8 1.8L11 11l3.8-1.8Z"/>
</svg>
"""

TARGET_ICON_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6" fill="currentColor"/>
</svg>
"""

INFO_ICON_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16.5"/><circle cx="12" cy="7.7" r="0.9" fill="currentColor" stroke="none"/>
</svg>
"""

MAP_PIN_ICON_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 21s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12Z"/><circle cx="12" cy="9" r="2.4"/>
</svg>
"""

SENSE_MARK_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/>
    <circle cx="12" cy="12" r="4.5"/>
    <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>
    <line x1="12" y1="1" x2="12" y2="3.2"/>
</svg>
"""

_RAIL_PATTERN_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="52" height="120" viewBox="0 0 52 120">
    <circle cx="26" cy="14" r="1.6" fill="white" fill-opacity="0.25"/>
    <circle cx="12" cy="40" r="1.1" fill="white" fill-opacity="0.16"/>
    <circle cx="40" cy="40" r="1.1" fill="white" fill-opacity="0.16"/>
    <circle cx="26" cy="64" r="1.6" fill="white" fill-opacity="0.25"/>
    <circle cx="12" cy="90" r="1.1" fill="white" fill-opacity="0.16"/>
    <circle cx="40" cy="90" r="1.1" fill="white" fill-opacity="0.16"/>
    <circle cx="26" cy="112" r="1.6" fill="white" fill-opacity="0.22"/>
</svg>
"""
RAIL_PATTERN_DATA_URI = "data:image/svg+xml," + urllib.parse.quote(_RAIL_PATTERN_SVG.strip())

DIRECTION_ARROWS = {"LEFT": "←", "CENTER": "●", "RIGHT": "→"}
DIRECTION_CLASS = {
    "LEFT": "ns-direction-left",
    "CENTER": "ns-direction-center",
    "RIGHT": "ns-direction-right",
}


def render_brand():
    """Draws the NavSense AI wordmark + tagline."""
    render_html(f"""
    <div class="ns-brand">
        <div class="ns-logo-mark">{EYE_ICON_SVG}</div>
        <div class="ns-brand-text">
            <h1>NavSense AI</h1>
            <p>Navigation assistance</p>
        </div>
    </div>
    """)


def render_navbar():
    """Draws the horizontal navigation bar."""
    cols = st.columns([2.6, 1, 1, 1, 0.5], gap="small")

    with cols[0]:
        render_brand()

    for i, item in enumerate(NAV_ITEMS):
        with cols[i + 1]:
            if st.session_state.page == item:
                render_html(f'<div class="ns-nav-active">{item}</div>')
            else:
                if st.button(item, key=f"nav_{item}"):
                    st.session_state.page = item

    with cols[4]:
        if st.button("⚙", key="nav_settings", help="Camera settings"):
            st.session_state.page = "Live Detection"
            st.session_state.show_settings = True

    render_html('<div class="ns-navbar-divider"></div>')


def render_status_pill(placeholder, state: str):
    """Writes a small status pill into placeholder."""
    if state == "running":
        dot_color, label = COLOR_SUCCESS, "Vision system active"
    elif state == "error":
        dot_color, label = COLOR_ERROR, "Camera unavailable"
    else:
        dot_color, label = COLOR_TEXT_MUTED, "Vision system idle"

    render_html_into(placeholder, f"""
    <div class="ns-status-wrap">
        <div class="ns-status-pill">
            <span class="ns-dot" style="background:{dot_color};"></span>
            <span>{label}</span>
        </div>
    </div>
    """)


def build_detections_html(rows):
    """Builds HTML for detected objects list."""
    if not rows:
        return f"""
        <div class="ns-empty">
            <div class="ns-empty-icon">{LIST_ICON_SVG}</div>
            <p class="ns-empty-title">Nothing detected yet</p>
            <p class="ns-empty-subtitle">Objects detected by the camera will appear here in real-time.</p>
        </div>
        """

    html = ""
    for row in rows:
        direction = row["direction"]
        arrow = DIRECTION_ARROWS.get(direction, "")
        badge_class = DIRECTION_CLASS.get(direction, "ns-direction-center")
        html += f"""
        <div class="ns-object-row">
            <div>
                <div class="ns-object-name">{row['class_name'].capitalize()}</div>
                <div class="ns-object-confidence">{row['confidence'] * 100:.0f}% confidence</div>
            </div>
            <div class="ns-direction-badge {badge_class}">{arrow} {direction.capitalize()}</div>
        </div>
        """
    return html


def build_guidance_sentence(rows):
    """Turns detection rows into plain-language summary."""
    if not rows:
        return "No objects are currently detected. The space ahead appears clear based on the last update."

    by_direction = {"CENTER": [], "LEFT": [], "RIGHT": []}
    for row in rows:
        by_direction[row["direction"]].append(row["class_name"])

    phrases = []
    if by_direction["CENTER"]:
        names = ", ".join(sorted(set(by_direction["CENTER"])))
        phrases.append(f"Directly ahead: {names}.")
    if by_direction["LEFT"]:
        names = ", ".join(sorted(set(by_direction["LEFT"])))
        phrases.append(f"To your left: {names}.")
    if by_direction["RIGHT"]:
        names = ", ".join(sorted(set(by_direction["RIGHT"])))
        phrases.append(f"To your right: {names}.")

    return " ".join(phrases)


@st.cache_resource
def load_detector(confidence_threshold: float) -> ObjectDetector:
    return ObjectDetector(model_path="yolov8n.pt", confidence_threshold=confidence_threshold)


def draw_detection(frame, class_name, confidence, direction, box):
    """Draws bounding box + label directly on the frame in high-visibility marigold."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR_BGR, 3)

    label = f"{class_name} {confidence:.2f} {direction}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y1 = max(y1 - text_h - 8, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 8, y1), BOX_COLOR_BGR, -1)
    cv2.putText(
        frame, label, (x1 + 4, y1 - 5 if y1 - 5 > 0 else y1 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Page: Live Detection
# ---------------------------------------------------------------------------
def render_live_detection_page():
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        render_html('<div class="ns-page-title">Live Detection</div>')
        render_html('<div class="ns-page-subtitle">Real-time camera feed with object and direction detection.</div>')
    with top_col2:
        status_placeholder = st.empty()

    col_camera, col_detections = st.columns([2.2, 1], gap="large")

    with col_camera:
        render_html('<div class="ns-section-label">Live camera feed</div>')
        camera_container = st.container(border=True)
        with camera_container:
            video_placeholder = st.empty()
            camera_on = st.toggle("Turn on camera", key="camera_on")

        with st.expander("Camera settings", expanded=st.session_state.show_settings):
            camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
            confidence_threshold = st.slider("Detection sensitivity", 0.1, 0.9, 0.5, 0.05)

    with col_detections:
        render_html('<div class="ns-section-label">Detected objects</div>')
        detections_container = st.container(border=True)
        with detections_container:
            detections_placeholder = st.empty()

    if camera_on:
        try:
            detector = load_detector(confidence_threshold)
        except Exception as e:
            render_status_pill(status_placeholder, "error")
            with camera_container:
                st.error(f"Could not load the detection model: {e}")
            st.stop()

        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

        if not cap.isOpened():
            render_status_pill(status_placeholder, "error")
            with camera_container:
                st.error(
                    "Could not open the webcam. Check that it's connected, not in "
                    "use by another app, and that camera permissions are enabled."
                )
            st.stop()

        render_status_pill(status_placeholder, "running")

        while camera_on:
            ret, frame = cap.read()

            if not ret or frame is None:
                render_status_pill(status_placeholder, "error")
                with camera_container:
                    st.error("Lost connection to the webcam. Turn the camera off and on again.")
                break

            frame_width = frame.shape[1]
            detections = detector.detect(frame)

            rows = []
            for det in detections:
                direction = get_direction(det["box"], frame_width)
                draw_detection(frame, det["class_name"], det["confidence"], direction, det["box"])
                rows.append({
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "direction": direction,
                })

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            render_html_into(detections_placeholder, build_detections_html(rows))

            st.session_state.last_detections = rows
            st.session_state.last_updated = time.time()

        cap.release()

    else:
        render_status_pill(status_placeholder, "idle")
        render_html_into(video_placeholder, f"""
        <div class="ns-empty">
            <div class="ns-empty-icon">{CAMERA_ICON_SVG}</div>
            <p class="ns-empty-title">Camera is off</p>
            <p class="ns-empty-subtitle">Turn it on above to start real-time object and direction detection.</p>
        </div>
        """)
        render_html_into(detections_placeholder, build_detections_html([]))


# ---------------------------------------------------------------------------
# Page: Guidance
# ---------------------------------------------------------------------------
def render_guidance_page():
    render_html('<div class="ns-page-title">Guidance</div>')
    render_html('<div class="ns-page-subtitle">A plain-language summary of what NavSense AI currently sees.</div>')

    rows = st.session_state.get("last_detections", [])
    last_updated = st.session_state.get("last_updated")

    sentence = build_guidance_sentence(rows)

    if last_updated:
        age_seconds = time.time() - last_updated
        if st.session_state.get("camera_on") and age_seconds < 5:
            meta = "Updating live from Live Detection."
        else:
            meta = f"Last updated {int(age_seconds)} seconds ago. Open Live Detection and turn the camera on to refresh this."
    else:
        meta = "Turn the camera on from Live Detection to start receiving guidance."

    render_html(f"""
    <div class="ns-guidance-panel">
        <div class="ns-guidance-icon">{COMPASS_ICON_SVG}</div>
        <div>
            <p class="ns-guidance-text">{sentence}</p>
            <div class="ns-guidance-meta">{meta}</div>
        </div>
    </div>
    """)

    render_html('<div class="ns-section-label">Detected objects</div>')
    with st.container(border=True):
        render_html(build_detections_html(rows))


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def render_about_page():
    render_html('<div class="ns-page-title">About</div>')
    render_html('<div class="ns-page-subtitle">What NavSense AI is and how it works.</div>')

    render_html(f"""
    <div class="ns-about-panel">
        <p>
            NavSense AI is a navigation assistance system designed to help visually
            impaired users understand what's around them. It uses a regular camera
            feed to notice nearby objects and describe roughly where they are - to
            the left, straight ahead, or to the right.
        </p>

        <div class="ns-about-heading">{TARGET_ICON_SVG}<h3>How it works</h3></div>
        <div class="ns-about-steps">
            <div class="ns-about-step">
                <div class="ns-about-step-icon">{CAMERA_ICON_SVG}</div>
                <h4>See</h4>
                <p>The camera captures a live view of what's ahead.</p>
            </div>
            <div class="ns-about-step">
                <div class="ns-about-step-icon">{MAP_PIN_ICON_SVG}</div>
                <h4>Detect</h4>
                <p>Nearby objects are identified in real time - people, furniture, doorways, and more.</p>
            </div>
            <div class="ns-about-step">
                <div class="ns-about-step-icon">{COMPASS_ICON_SVG}</div>
                <h4>Locate</h4>
                <p>Each object is placed to your left, center, or right.</p>
            </div>
        </div>

        <div class="ns-about-heading">{INFO_ICON_SVG}<h3>Where this stands today</h3></div>
        <p class="ns-about-note">
            This early version focuses on getting real-time object and direction
            awareness right. Later versions aim to add richer guidance, such as
            understanding how far away something is and speaking guidance aloud.
        </p>
    </div>
    """)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Live Detection"
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

inject_css()
render_html(f"""
<div class="ns-edge-rail">
    <div class="ns-edge-mark">{SENSE_MARK_SVG}</div>
    <div class="ns-edge-label">NAVSENSE AI</div>
</div>
""")
render_navbar()

if st.session_state.page == "Live Detection":
    render_live_detection_page()
elif st.session_state.page == "Guidance":
    render_guidance_page()
elif st.session_state.page == "About":
    render_about_page()
