"""
streamlit.py

Streamlit UI for the NavSense AI prototype.

Redesigned with a blooming, botanical-inspired interface (warm ivory, sage green, leaf green,
marigold yellow, apricot, soft lavender), vertical left-side navigation sidebar with 8 dedicated
sections, rich interactive feature cards, and complete preservation of all YOLOv8, MiDaS, distance,
prioritization, navigation, and voice assistance logic.
"""

import textwrap
import time
import urllib.parse

import cv2
import streamlit as st

from detection import ObjectDetector
from depth import DepthEstimator         
from direction import get_direction
from distance import ObstacleDistanceEstimator 
from priority import ObstaclePrioritizer           
from navigation import NavigationAdvisor
from voice import VoiceGuide
from html import escape


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NavSense AI",
    page_icon="🌿",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Design tokens - Botanical Palette
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
COLOR_PEACH = "#E8A87C"         # Peach botanical accent
COLOR_LAVENDER = "#9A8C98"      # Subtle lavender tone

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
# Global styling & Botanical CSS
# ---------------------------------------------------------------------------
def inject_css():
    render_html(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
        html, body, [class*="css"] {{
            font-family: 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: {COLOR_TEXT};
        }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {COLOR_BG};
        }}

        /* Hide Streamlit default chrome */
        #MainMenu, footer {{ visibility: hidden; }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1240px;
        }}

        /* ---------- Sidebar Styling ---------- */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_SURFACE};
            border-right: 2px solid {COLOR_BORDER};
            padding-top: 1.5rem;
        }}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            gap: 0.5rem;
        }}

        /* Style Streamlit Native Bordered Containers with High Contrast Border */
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            background-color: {COLOR_SURFACE} !important;
            border: 2px solid {COLOR_BORDER} !important;
            border-radius: 14px !important;
            box-shadow: 0 14px 34px rgba(63, 79, 48, 0.08);
            transition: all 0.3s ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
            border-color: {COLOR_ACCENT} !important;
            box-shadow: 0 18px 42px rgba(63, 107, 69, 0.14);
        }}

        /* Distinct font for expander */
        [data-testid="stExpander"] summary p {{
            font-family: 'Manrope', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: {COLOR_ACCENT} !important;
        }}

        label, [data-testid="stWidgetLabel"] p {{
            color: {COLOR_TEXT} !important;
            font-weight: 600 !important;
        }}

        /* ---------- Brand & Header ---------- */
        .ns-brand {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.5rem 0.2rem 1.2rem 0.2rem;
            border-bottom: 1px solid {COLOR_BORDER};
            margin-bottom: 1.2rem;
        }}
        .ns-logo-mark {{
            width: 42px;
            height: 42px;
            flex-shrink: 0;
            border-radius: 50%;
            background: {COLOR_ACCENT_SOFT};
            color: {COLOR_ACCENT};
            border: 2px solid {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(63, 107, 69, 0.2);
            position: relative;
        }}
        .ns-logo-mark::after {{
            content: '';
            position: absolute;
            width: 10px;
            height: 10px;
            right: -2px;
            top: -2px;
            border-radius: 50%;
            background: {COLOR_AMBER};
            border: 2px solid {COLOR_SURFACE};
        }}
        .ns-brand-text h1 {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            color: {COLOR_TEXT};
        }}
        .ns-brand-text p {{
            font-size: 0.78rem;
            margin: 0;
            color: {COLOR_TEXT_MUTED};
        }}

        /* ---------- Hero & Cards ---------- */
        .ns-hero-card {{
            background: linear-gradient(135deg, {COLOR_SURFACE} 0%, {COLOR_SURFACE_ALT} 100%);
            border: 2px solid {COLOR_BORDER};
            border-radius: 20px;
            padding: 2.5rem 2.2rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 20px 50px rgba(63, 79, 48, 0.12);
            position: relative;
            overflow: hidden;
        }}
        .ns-hero-card::before {{
            content: '';
            position: absolute;
            top: -50px;
            right: -50px;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(200, 120, 32, 0.15) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }}
        .ns-hero-title {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: clamp(2rem, 3.8vw, 2.8rem);
            font-weight: 800;
            line-height: 1.15;
            color: {COLOR_TEXT};
            margin-bottom: 1rem;
        }}
        .ns-hero-subtitle {{
            font-size: 1.05rem;
            line-height: 1.6;
            color: {COLOR_TEXT_MUTED};
            max-width: 42rem;
            margin-bottom: 1.5rem;
        }}

        .ns-feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.2rem;
            margin-bottom: 1.8rem;
        }}
        .ns-feature-card {{
            background: {COLOR_SURFACE};
            border: 1.5px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 0 8px 24px rgba(63, 79, 48, 0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .ns-feature-card:hover {{
            transform: translateY(-3px);
            border-color: {COLOR_ACCENT};
        }}
        .ns-feature-icon {{
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: {COLOR_ACCENT_SOFT};
            color: {COLOR_ACCENT};
            border: 1px solid {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }}
        .ns-feature-card h3 {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: {COLOR_TEXT};
            margin: 0 0 0.5rem 0;
        }}
        .ns-feature-card p {{
            font-size: 0.9rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0;
            line-height: 1.5;
        }}

        /* ---------- Status pill ---------- */
        .ns-status-wrap {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.5rem;
        }}
        .ns-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            border: 1.5px solid {COLOR_BORDER};
            background: {COLOR_SURFACE};
            font-size: 0.85rem;
            font-weight: 600;
            color: {COLOR_TEXT};
            box-shadow: 0 4px 12px rgba(63, 79, 48, 0.08);
        }}
        .ns-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
            animation: pulse-dot 2s infinite;
        }}
        @keyframes pulse-dot {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.15); opacity: 1; box-shadow: 0 0 8px {COLOR_SUCCESS}; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        /* ---------- Section labels / page headers ---------- */
        .ns-section-label {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {COLOR_ACCENT};
            margin-bottom: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .ns-page-title {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 800;
            color: {COLOR_TEXT};
            margin: 0 0 0.4rem 0;
        }}
        .ns-page-subtitle {{
            font-size: 1rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0 0 1.5rem 0;
        }}

        /* ---------- Empty states ---------- */
        .ns-empty {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            padding: 4rem 1.5rem;
            text-align: center;
        }}
        .ns-empty-icon {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: {COLOR_ACCENT_SOFT};
            border: 2px solid {COLOR_BORDER};
            color: {COLOR_ACCENT};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .ns-empty-title {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: {COLOR_TEXT};
            margin: 0;
        }}
        .ns-empty-subtitle {{
            font-size: 0.9rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0;
            max-width: 28em;
        }}

        /* ---------- Detected object list & priorities ---------- */
        .ns-object-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            padding: 0.85rem 0.8rem;
            border-bottom: 1px solid {COLOR_SURFACE_ALT};
            border-radius: 8px;
            transition: background 0.15s ease;
        }}
        .ns-object-row:hover {{
            background: rgba(63, 107, 69, 0.04);
        }}
        .ns-object-row:last-child {{
            border-bottom: none;
        }}
        .ns-object-name {{
            font-size: 0.98rem;
            font-weight: 700;
            color: {COLOR_TEXT};
        }}
        .ns-object-confidence {{
            font-size: 0.82rem;
            color: {COLOR_TEXT_MUTED};
            margin-top: 0.15rem;
        }}
        .ns-direction-badge {{
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            letter-spacing: 0.02em;
        }}
        .ns-direction-left   {{ background: rgba(154, 140, 152, 0.15); color: #633D50; border: 1px solid #8A6575; }}
        .ns-direction-center {{ background: rgba(86, 128, 68, 0.15); color: {COLOR_SUCCESS}; border: 1px solid {COLOR_SUCCESS}; }}
        .ns-direction-right  {{ background: rgba(200, 120, 32, 0.15); color: {COLOR_AMBER}; border: 1px solid {COLOR_AMBER}; }}
        .ns-distance-near {{ background: rgba(167, 68, 46, 0.15); color: {COLOR_ERROR}; border: 1px solid {COLOR_ERROR}; }}
        .ns-distance-medium {{ background: rgba(200, 120, 32, 0.15); color: {COLOR_AMBER}; border: 1px solid {COLOR_AMBER}; }}
        .ns-distance-far {{ background: rgba(86, 128, 68, 0.15); color: {COLOR_SUCCESS}; border: 1px solid {COLOR_SUCCESS}; }}

        .ns-priority-card {{
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: 1rem 1.1rem; margin-bottom: 0.8rem; border-radius: 12px;
            background: {COLOR_SURFACE_ALT}; border: 1.5px solid {COLOR_BORDER};
        }}
        .ns-priority-label {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: {COLOR_TEXT_MUTED}; }}
        .ns-priority-name {{ font-family: 'Fraunces', Georgia, serif; font-size: 1.15rem; font-weight: 700; color: {COLOR_TEXT}; margin-top: 0.1rem; }}
        .ns-priority-detail {{ font-size: 0.85rem; color: {COLOR_TEXT_MUTED}; margin-top: 0.15rem; }}
        .ns-risk-high {{ background: rgba(167, 68, 46, 0.15); color: {COLOR_ERROR}; border: 1.5px solid {COLOR_ERROR}; }}
        .ns-risk-medium {{ background: rgba(200, 120, 32, 0.15); color: {COLOR_AMBER}; border: 1.5px solid {COLOR_AMBER}; }}
        .ns-risk-low {{ background: rgba(86, 128, 68, 0.15); color: {COLOR_SUCCESS}; border: 1.5px solid {COLOR_SUCCESS}; }}

        /* ---------- Panels & Workflow Steps ---------- */
        .ns-panel {{
            background: {COLOR_SURFACE};
            border: 2px solid {COLOR_BORDER};
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 36px rgba(63, 79, 48, 0.08);
        }}
        .ns-panel p {{
            font-size: 1rem;
            line-height: 1.7;
            color: {COLOR_TEXT};
            margin: 0 0 1rem 0;
        }}
        .ns-workflow-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        .ns-workflow-step {{
            background: {COLOR_SURFACE_ALT};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }}
        .ns-workflow-num {{
            font-size: 0.75rem;
            font-weight: 700;
            color: {COLOR_ACCENT};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.3rem;
        }}
        .ns-workflow-title {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1rem;
            font-weight: 700;
            color: {COLOR_TEXT};
            margin-bottom: 0.4rem;
        }}
        .ns-workflow-desc {{
            font-size: 0.82rem;
            color: {COLOR_TEXT_MUTED};
            line-height: 1.4;
            margin: 0;
        }}

        /* ---------- Footer ---------- */
        .ns-footer {{
            margin-top: 3rem;
            padding: 1.5rem 0;
            border-top: 1px solid {COLOR_BORDER};
            text-align: center;
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.05rem;
            font-style: italic;
            color: {COLOR_ACCENT};
            letter-spacing: 0.02em;
        }}

        /* Native Button styling tweak */
        .stButton>button {{
            border-radius: 10px;
            border: 1.5px solid {COLOR_BORDER};
            background: {COLOR_SURFACE};
            color: {COLOR_TEXT};
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background: {COLOR_ACCENT_SOFT};
            border-color: {COLOR_ACCENT};
            color: {COLOR_ACCENT};
        }}
    </style>
    """)


# ---------------------------------------------------------------------------
# SVG Icons
# ---------------------------------------------------------------------------
ICON_HOME = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>'
ICON_CAMERA = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h3l2-2h8l2 2h3v12H3z"/><circle cx="12" cy="13" r="3.5"/></svg>'
ICON_GUIDANCE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M14.8 9.2 13 13l-3.8 1.8L11 11l3.8-1.8Z"/></svg>'
ICON_ENV = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>'
ICON_HISTORY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>'
ICON_WORKFLOW = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>'
ICON_SAFETY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
ICON_ABOUT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><circle cx="12" cy="7.5" r="0.8" fill="currentColor"/></svg>'

DIRECTION_ARROWS = {"LEFT": "←", "CENTER": "●", "RIGHT": "→"}
DIRECTION_CLASS = {
    "LEFT": "ns-direction-left",
    "CENTER": "ns-direction-center",
    "RIGHT": "ns-direction-right",
}
NAV_RISK_CLASS = {
    "Stop": "ns-risk-high",
    "Slow Down": "ns-risk-medium",
    "Keep Left": "ns-risk-medium",
    "Keep Right": "ns-risk-medium",
    "Path Clear": "ns-risk-low",
}
NAV_ARROWS = {"Stop": "■", "Slow Down": "▼", "Keep Left": "←", "Keep Right": "→", "Path Clear": "▲"}
VOICE_KIND = {
    0: ("Emergency", "ns-risk-high"),
    1: ("Steering", "ns-risk-medium"),
    2: ("Warning", "ns-risk-medium"),
    3: ("Info", "ns-risk-low"),
}


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        render_html(f"""
        <div class="ns-brand">
            <div class="ns-logo-mark">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
            </div>
            <div class="ns-brand-text">
                <h1>NavSense AI</h1>
                <p>Botanical Vision Assistant</p>
            </div>
        </div>
        """)

        pages = {
            "Home": ICON_HOME,
            "Live Detection": ICON_CAMERA,
            "Navigation Guidance": ICON_GUIDANCE,
            "Environment Overview": ICON_ENV,
            "Detection History": ICON_HISTORY,
            "How It Works": ICON_WORKFLOW,
            "Accessibility & Safety": ICON_SAFETY,
            "About NavSense AI": ICON_ABOUT,
        }

        for page_name, icon_svg in pages.items():
            is_active = st.session_state.page == page_name
            btn_type = "primary" if is_active else "secondary"
            if st.sidebar.button(f"{page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.page = page_name
                st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"<p style='font-size: 0.78rem; color: {COLOR_TEXT_MUTED}; text-align: center;'>🌿 Powered by YOLOv8 & MiDaS</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Status Pill Helper
# ---------------------------------------------------------------------------
def render_status_pill(placeholder, state: str):
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


# ---------------------------------------------------------------------------
# Component Builders (HTML helpers)
# ---------------------------------------------------------------------------
def build_priority_card(row):
    risk = row["risk_level"]
    parts = [row["direction"].capitalize()]
    if row.get("distance"):
        parts.insert(0, row["distance"])
    return f"""
    <div class="ns-priority-card">
        <div>
            <div class="ns-priority-label">Highest priority obstacle</div>
            <div class="ns-priority-name">{row['class_name'].capitalize()}</div>
            <div class="ns-priority-detail">{' · '.join(parts)}</div>
        </div>
        <div class="ns-direction-badge ns-risk-{risk.lower()}">{risk} risk</div>
    </div>
    """


def build_navigation_html(nav):
    if not nav:
        return """
        <div class="ns-priority-card">
            <div>
                <div class="ns-priority-label">Navigation Guidance</div>
                <div class="ns-priority-name">Waiting for camera</div>
                <div class="ns-priority-detail">Real-time directions appear here once detection starts.</div>
            </div>
        </div>
        """
    instruction = nav["instruction"]
    return f"""
    <div class="ns-priority-card">
        <div>
            <div class="ns-priority-label">Navigation Action</div>
            <div class="ns-priority-name">{instruction}</div>
            <div class="ns-priority-detail">{nav['reason']}</div>
        </div>
        <div class="ns-direction-badge {NAV_RISK_CLASS.get(instruction, 'ns-risk-low')}">{NAV_ARROWS.get(instruction, '')} {instruction}</div>
    </div>
    """


def build_voice_html(voice):
    def card(name, detail, badge=""):
        return f"""
        <div class="ns-priority-card">
            <div>
                <div class="ns-priority-label">Voice Guidance Status</div>
                <div class="ns-priority-name">{name}</div>
                <div class="ns-priority-detail">{detail}</div>
            </div>
            {badge}
        </div>
        """

    if not voice:
        return card("Waiting for camera", "Spoken instructions appear here once active.")
    if not voice["available"]:
        return card("Voice unavailable", escape(voice.get("error") or "Speech engine not loaded."))
    if not voice["enabled"]:
        return card("Voice muted", "Turn on 'Voice guidance' in camera settings.")

    if voice.get("error"):
        return card("Voice error", escape(voice["error"]))
    history = voice["history"]
    if not history:
        return card("Listening…", "No announcements spoken yet.")

    latest = history[0]
    label, css = VOICE_KIND.get(latest["priority"], ("Info", "ns-risk-low"))
    age = max(0, int(time.time() - latest["time"]))
    when = "just now" if age < 2 else f"{age}s ago"
    earlier = " · ".join(escape(h["text"]) for h in history[1:4])
    detail = when + (f" · Previous: {earlier}" if earlier else "")
    badge = f'<div class="ns-direction-badge {css}">{label}</div>'
    return card(f"“{escape(latest['text'])}”", detail, badge)


def build_detections_html(rows):
    if not rows:
        return f"""
        <div class="ns-empty">
            <div class="ns-empty-icon">🌿</div>
            <p class="ns-empty-title">No objects detected yet</p>
            <p class="ns-empty-subtitle">Objects identified by the camera feed will display here in real time.</p>
        </div>
        """

    out = ""
    top = next((r for r in rows if r.get("rank") == 1), None)
    if top:
        out += build_priority_card(top)
    for row in rows:
        direction = row["direction"]
        arrow = DIRECTION_ARROWS.get(direction, "")
        badge_class = DIRECTION_CLASS.get(direction, "ns-direction-center")
        distance = row.get("distance")
        distance_badge = (
            f'<div class="ns-direction-badge ns-distance-{distance.lower()}">{distance}</div>'
            if distance else ""
        )
        risk = row.get("risk_level")
        risk_badge = (
            f'<div class="ns-direction-badge ns-risk-{risk.lower()}">{risk}</div>'
            if risk else ""
        )
        score = row.get("score")
        depth_text = f"depth score {score:.2f}" if score is not None else "depth n/a"
        rank_prefix = f"#{row['rank']} " if row.get("rank") else ""
        out += f"""
        <div class="ns-object-row">
            <div>
                <div class="ns-object-name">{rank_prefix}{row['class_name'].capitalize()}</div>
                <div class="ns-object-confidence">{row['confidence'] * 100:.0f}% confidence · {depth_text}</div>
            </div>
            <div style="display:flex; gap:0.4rem; align-items:center; flex-wrap:wrap; justify-content:flex-end;">
                {risk_badge}
                {distance_badge}
                <div class="ns-direction-badge {badge_class}">{arrow} {direction.capitalize()}</div>
            </div>
        </div>
        """
    return out


def build_guidance_sentence(rows):
    if not rows:
        return "No objects are currently detected. The path ahead appears peaceful and clear."

    top = next((r for r in rows if r.get("rank") == 1), None)
    prefix = ""
    if top:
        where = {"CENTER": "directly ahead", "LEFT": "to your left", "RIGHT": "to your right"}[top["direction"]]
        dist = f"{top['distance'].lower()} " if top.get("distance") else ""
        prefix = f"Highest priority: {dist}{top['class_name']} {where}. "
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

    return prefix + " ".join(phrases)


@st.cache_resource
def load_detector(confidence_threshold: float) -> ObjectDetector:
    return ObjectDetector(model_path="yolov8n.pt", confidence_threshold=confidence_threshold)

@st.cache_resource(show_spinner="Loading botanical depth model…")
def load_depth_estimator() -> DepthEstimator:
    return DepthEstimator()

@st.cache_resource
def load_voice_guide() -> VoiceGuide:
    return VoiceGuide()

DEPTH_EVERY_N_FRAMES = 2


def draw_detection(frame, class_name, confidence, direction, box, distance=None, is_priority=False):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR_BGR, 6 if is_priority else 3)
    label = f"{class_name} {confidence:.2f} {direction}"
    if distance:
        label += f" {distance}"
    if is_priority:
        label += " PRIORITY"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y1 = max(y1 - text_h - 8, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 8, y1), BOX_COLOR_BGR, -1)
    cv2.putText(
        frame, label, (x1 + 4, y1 - 5 if y1 - 5 > 0 else y1 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Pages Implementation
# ---------------------------------------------------------------------------

def render_home_page():
    render_html(f"""
    <div class="ns-hero-card">
        <div class="ns-hero-title">Navigate Your World with Confidence</div>
        <div class="ns-hero-subtitle">
            NavSense AI is a botanical-inspired spatial vision assistant combining YOLOv8 object detection
            and MiDaS depth estimation to bring clarity, safety, and independence to every step.
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        render_html(f"""
        <div class="ns-feature-card">
            <div class="ns-feature-icon">🌿</div>
            <h3>Botanical Clarity</h3>
            <p>Real-time visual and audio cues designed with calm warmth and zero cognitive overload.</p>
        </div>
        """)
    with col2:
        render_html(f"""
        <div class="ns-feature-card">
            <div class="ns-feature-icon">👁️</div>
            <h3>YOLOv8 + MiDaS</h3>
            <p>State-of-the-art computer vision estimating distances and threat priority instantly.</p>
        </div>
        """)
    with col3:
        render_html(f"""
        <div class="ns-feature-card">
            <div class="ns-feature-icon">🎙️</div>
            <h3>Voice Guidance</h3>
            <p>Clear spoken directions to guide your navigation effortlessly without screen glances.</p>
        </div>
        """)

    st.markdown("### ")
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("Start Live Detection ➔", use_container_width=True, type="primary"):
            st.session_state.page = "Live Detection"
            st.rerun()


def render_live_detection_page():
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        render_html('<div class="ns-page-title">Live Detection</div>')
        render_html('<div class="ns-page-subtitle">Real-time camera feed with object detection and depth mapping.</div>')
    with top_col2:
        status_placeholder = st.empty()

    col_camera, col_detections = st.columns([2.2, 1], gap="large")

    with col_camera:
        render_html('<div class="ns-section-label">Live Camera Feed</div>')
        camera_container = st.container(border=True)
        with camera_container:
            render_html('<div class="ns-object-confidence">Camera Feed</div>')
            video_placeholder = st.empty()
            render_html('<div style="margin-top: 1rem;"></div>')
            render_html('<div class="ns-object-confidence">Depth Map (bright = near)</div>')
            depth_placeholder = st.empty()
            camera_on = st.toggle("Turn on camera", key="camera_on")

        with st.expander("Camera & Model Settings", expanded=False):
            camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
            confidence_threshold = st.slider("Detection sensitivity", 0.1, 0.9, 0.5, 0.05)
            voice_on = st.toggle("Voice guidance", value=True, key="voice_on")

    with col_detections:
        navigation_placeholder = st.empty()
        voice_placeholder = st.empty()
        render_html('<div class="ns-section-label">Detected Objects</div>')
        detections_container = st.container(border=True)
        with detections_container:
            detections_placeholder = st.empty()

    if camera_on:
        try:
            detector = load_detector(confidence_threshold)
        except Exception as e:
            render_status_pill(status_placeholder, "error")
            with camera_container:
                st.error(f"Could not load detection model: {e}")
            st.stop()

        depth_estimator = None
        try:
            depth_estimator = load_depth_estimator()
        except Exception as e:
            depth_placeholder.warning(f"Depth model unavailable: {e}")

        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

        if not cap.isOpened():
            render_status_pill(status_placeholder, "error")
            with camera_container:
                st.error("Could not open webcam. Ensure it is connected and permissions are granted.")
            st.stop()

        render_status_pill(status_placeholder, "running")

        frame_count = 0
        depth_map = None
        distance_estimator = ObstacleDistanceEstimator()
        prioritizer = ObstaclePrioritizer()
        navigator = NavigationAdvisor()
        voice = load_voice_guide()
        voice.reset()
        voice.set_enabled(voice_on)

        while camera_on:
            ret, frame = cap.read()
            if not ret or frame is None:
                render_status_pill(status_placeholder, "error")
                with camera_container:
                    st.error("Lost webcam connection.")
                break

            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            if depth_estimator is not None and frame_count % DEPTH_EVERY_N_FRAMES == 0:
                depth_map = depth_estimator.estimate(frame)
                depth_rgb = depth_estimator.colorize(depth_map)
                depth_placeholder.image(depth_rgb, channels="RGB", use_container_width=True)
            frame_count += 1

            detections = detector.detect(frame)
            rows = []
            distances = distance_estimator.update(depth_map, detections)
            ranking = prioritizer.rank(detections, distance_estimator.last_scores, frame_width)
            rank_info = {r["index"]: r for r in ranking}
            nav = navigator.advise(
                detections, distances, distance_estimator.last_scores,
                frame_width, frame_height, ranking,
            )

            scores = distance_estimator.last_scores
            for i, (det, distance) in enumerate(zip(detections, distances)):
                direction = get_direction(det["box"], frame_width)
                info = rank_info.get(i)
                draw_detection(
                    frame, det["class_name"], det["confidence"], direction, det["box"],
                    distance, is_priority=bool(info and info["rank"] == 1),
                )
                rows.append({
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "direction": direction,
                    "distance": distance,
                    "score": scores[i] if i < len(scores) else None,
                    "rank": info["rank"] if info else None,
                    "risk_level": info["risk_level"] if info else None,
                })

            rows.sort(key=lambda r: r["rank"] if r["rank"] is not None else 999)
            voice.update(nav, rows)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            render_html_into(navigation_placeholder, build_navigation_html(nav))
            voice_state = voice.snapshot()
            render_html_into(voice_placeholder, build_voice_html(voice_state))
            render_html_into(detections_placeholder, build_detections_html(rows))

            st.session_state.last_detections = rows
            st.session_state.last_navigation = nav
            st.session_state.last_updated = time.time()
            st.session_state.last_voice = voice_state

        cap.release()
    else:
        render_status_pill(status_placeholder, "idle")
        render_html_into(video_placeholder, f"""
        <div class="ns-empty">
            <div class="ns-empty-icon">📷</div>
            <p class="ns-empty-title">Camera is offline</p>
            <p class="ns-empty-subtitle">Toggle "Turn on camera" above to start live detection.</p>
        </div>
        """)
        render_html_into(navigation_placeholder, build_navigation_html(None))
        render_html_into(detections_placeholder, build_detections_html([]))
        voice = load_voice_guide()
        voice.reset()
        render_html_into(voice_placeholder, build_voice_html(voice.snapshot()))


def render_guidance_page():
    render_html('<div class="ns-page-title">Navigation Guidance</div>')
    render_html('<div class="ns-page-subtitle">Actionable instructions and spatial audio status for safe movement.</div>')

    rows = st.session_state.get("last_detections", [])
    last_updated = st.session_state.get("last_updated")
    sentence = build_guidance_sentence(rows)

    if last_updated:
        age_seconds = time.time() - last_updated
        meta = "Live update active." if st.session_state.get("camera_on") and age_seconds < 5 else f"Last updated {int(age_seconds)}s ago."
    else:
        meta = "Turn on the camera in Live Detection to receive real-time guidance."

    render_html(f"""
    <div class="ns-panel">
        <div style="font-size: 0.8rem; font-weight: 700; color: {COLOR_ACCENT}; text-transform: uppercase; margin-bottom: 0.5rem;">Spatial Summary</div>
        <p style="font-size: 1.15rem; font-weight: 500; margin-bottom: 0.8rem;">{sentence}</p>
        <div style="font-size: 0.82rem; color: {COLOR_TEXT_MUTED};">{meta}</div>
    </div>
    """)

    render_html(build_navigation_html(st.session_state.get("last_navigation")))
    render_html(build_voice_html(st.session_state.get("last_voice")))

    render_html('<div class="ns-section-label">Detected Obstacles</div>')
    with st.container(border=True):
        render_html(build_detections_html(rows))


def render_environment_overview_page():
    render_html('<div class="ns-page-title">Environment Overview</div>')
    render_html('<div class="ns-page-subtitle">Obstacle distribution and spatial analytics based on current camera telemetry.</div>')

    rows = st.session_state.get("last_detections", [])
    if not rows:
        render_html(f"""
        <div class="ns-panel">
            <div class="ns-empty">
                <div class="ns-empty-icon">📊</div>
                <p class="ns-empty-title">No telemetry data available</p>
                <p class="ns-empty-subtitle">Activate Live Detection to generate real-time environmental distribution charts.</p>
            </div>
        </div>
        """)
        return

    # Compute stats
    total_objs = len(rows)
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    dir_counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    for r in rows:
        if r.get("risk_level"):
            risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1
        if r.get("direction"):
            dir_counts[r["direction"]] = dir_counts.get(r["direction"], 0) + 1

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        render_html(f"""
        <div class="ns-panel" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; text-transform: uppercase;">Total Detected</div>
            <div style="font-family: 'Fraunces', Georgia, serif; font-size: 2.2rem; font-weight: 800; color: {COLOR_ACCENT}; margin: 0.3rem 0;">{total_objs}</div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_MUTED};">Active objects in view</div>
        </div>
        """)
    with col2:
        render_html(f"""
        <div class="ns-panel" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; text-transform: uppercase;">High Risk Threats</div>
            <div style="font-family: 'Fraunces', Georgia, serif; font-size: 2.2rem; font-weight: 800; color: {COLOR_ERROR}; margin: 0.3rem 0;">{risk_counts.get('High', 0)}</div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_MUTED};">Requiring immediate action</div>
        </div>
        """)
    with col3:
        render_html(f"""
        <div class="ns-panel" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; text-transform: uppercase;">Center Obstacles</div>
            <div style="font-family: 'Fraunces', Georgia, serif; font-size: 2.2rem; font-weight: 800; color: {COLOR_AMBER}; margin: 0.3rem 0;">{dir_counts.get('CENTER', 0)}</div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_MUTED};">Directly in path</div>
        </div>
        """)

    render_html('<div class="ns-section-label">Directional Distribution</div>')
    col_chart1, col_chart2 = st.columns(2, gap="medium")
    with col_chart1:
        with st.container(border=True):
            st.bar_chart(dir_counts)
    with col_chart2:
        with st.container(border=True):
            st.bar_chart(risk_counts)


def render_detection_history_page():
    render_html('<div class="ns-page-title">Detection History</div>')
    render_html('<div class="ns-page-subtitle">Chronological event timeline of recent spatial detections.</div>')

    rows = st.session_state.get("last_detections", [])
    if not rows:
        render_html(f"""
        <div class="ns-panel">
            <div class="ns-empty">
                <div class="ns-empty-icon">⏳</div>
                <p class="ns-empty-title">No history recorded yet</p>
                <p class="ns-empty-subtitle">Events captured during live camera sessions will appear in this timeline.</p>
            </div>
        </div>
        """)
        return

    render_html(f"""
    <div class="ns-panel">
        <p style="font-weight: 600; margin-bottom: 1.2rem;">Recent Session Detections ({len(rows)} items logged):</p>
        {build_detections_html(rows)}
    </div>
    """)


def render_how_it_works_page():
    render_html('<div class="ns-page-title">How NavSense AI Works</div>')
    render_html('<div class="ns-page-subtitle">The seamless AI pipeline from optical capture to intelligent voice guidance.</div>')

    render_html(f"""
    <div class="ns-panel">
        <p>
            NavSense AI runs a real-time computer vision pipeline optimized for speed and reliability,
            transforming raw camera pixels into spatial understanding and accessible feedback.
        </p>

        <div class="ns-workflow-grid">
            <div class="ns-workflow-step">
                <div class="ns-workflow-num">Step 01</div>
                <div class="ns-workflow-title">Camera Feed</div>
                <p class="ns-workflow-desc">Captures high-framerate environmental video.</p>
            </div>
            <div class="ns-workflow-step">
                <div class="ns-workflow-num">Step 02</div>
                <div class="ns-workflow-title">YOLOv8</div>
                <p class="ns-workflow-desc">Detects and classifies objects with high precision.</p>
            </div>
            <div class="ns-workflow-step">
                <div class="ns-workflow-num">Step 03</div>
                <div class="ns-workflow-title">MiDaS Depth</div>
                <p class="ns-workflow-desc">Estimates relative depth and proximity.</p>
            </div>
            <div class="ns-workflow-step">
                <div class="ns-workflow-num">Step 04</div>
                <div class="ns-workflow-title">Prioritization</div>
                <p class="ns-workflow-desc">Ranks obstacles by risk and urgency.</p>
            </div>
            <div class="ns-workflow-step">
                <div class="ns-workflow-num">Step 05</div>
                <div class="ns-workflow-title">Voice Assistant</div>
                <p class="ns-workflow-desc">Announces clear spoken guidance.</p>
            </div>
        </div>
    </div>
    """)


def render_accessibility_safety_page():
    render_html('<div class="ns-page-title">Accessibility & Safety</div>')
    render_html('<div class="ns-page-subtitle">Committed to safe, responsible, and empowering navigation assistance.</div>')

    render_html(f"""
    <div class="ns-panel">
        <h3 style="font-family: 'Fraunces', Georgia, serif; color: {COLOR_ACCENT}; margin-top: 0;">Our Commitment to Safety</h3>
        <p>
            NavSense AI is designed as a supplementary spatial awareness tool. While advanced deep learning
            models (YOLOv8 and MiDaS) provide robust object detection and depth estimation, technology should
            always be used alongside traditional mobility aids such as white canes or guide dogs.
        </p>
        <h3 style="font-family: 'Fraunces', Georgia, serif; color: {COLOR_ACCENT}; margin-top: 1.5rem;">Responsible Usage Guidelines</h3>
        <p>
            - Ensure adequate lighting for optimal camera performance.\n
            - Maintain device charge and stable camera positioning.\n
            - Always rely on primary mobility instincts and physical cane feedback in high-traffic or hazardous environments.
        </p>
    </div>
    """)


def render_about_ai_page():
    render_html('<div class="ns-page-title">About NavSense AI</div>')
    render_html('<div class="ns-page-subtitle">Project vision, architecture, technology stack, and future roadmap.</div>')

    render_html(f"""
    <div class="ns-panel">
        <h3 style="font-family: 'Fraunces', Georgia, serif; color: {COLOR_ACCENT}; margin-top: 0;">Project Vision</h3>
        <p>
            NavSense AI bridges the gap between advanced artificial intelligence and assistive technology,
            creating an intuitive, warm, and joyful experience that empowers visually impaired individuals.
        </p>
        <h3 style="font-family: 'Fraunces', Georgia, serif; color: {COLOR_ACCENT}; margin-top: 1.5rem;">Technology Stack</h3>
        <p>
            - <strong>Python & Streamlit</strong>: Interactive, responsive user interface architecture.\n
            - <strong>YOLOv8</strong>: Ultra-fast object detection and classification.\n
            - <strong>MiDaS</strong>: Monocular depth estimation for spatial awareness.\n
            - <strong>OpenCV & pyttsx3</strong>: Real-time image processing and speech synthesis.
        </p>
    </div>
    """)


# ---------------------------------------------------------------------------
# Entry Point & State Initialization
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

inject_css()
render_sidebar()

# Route pages
page = st.session_state.page
if page == "Home":
    render_home_page()
elif page == "Live Detection":
    render_live_detection_page()
elif page == "Navigation Guidance":
    render_guidance_page()
elif page == "Environment Overview":
    render_environment_overview_page()
elif page == "Detection History":
    render_detection_history_page()
elif page == "How It Works":
    render_how_it_works_page()
elif page == "Accessibility & Safety":
    render_accessibility_safety_page()
elif page == "About NavSense AI":
    render_about_ai_page()

# Footer
render_html('<div class="ns-footer">Every step deserves confidence.</div>')