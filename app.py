import streamlit as st
from PIL import Image
from style_transfer import run_style_transfer, STYLE_PROMPTS
import io

st.set_page_config(page_title="AI Style Transfer", layout="wide")
st.title("🎨 AI Art Style Transfer")
st.markdown("Transforms your photo into actual artwork using Stable Diffusion.")

content_file = st.file_uploader("📷 Upload your photo", type=["jpg","jpeg","png"])

if content_file:
    content_img = Image.open(content_file).convert("RGB")
    st.image(content_img, caption="Your Photo", width=300)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    style_name = st.selectbox("🎨 Choose Art Style", list(STYLE_PROMPTS.keys()))

with col2:
    strength = st.slider(
        "Transformation Strength",
        0.4, 0.95, 0.75, 0.05,
        help="Higher = more artistic, less like original photo"
    )

with col3:
    guidance = st.slider(
        "Style Guidance",
        5.0, 20.0, 12.0, 0.5,
        help="Higher = follows the style more strictly"
    )

if st.button("✨ Generate Artwork", use_container_width=True):
    if not content_file:
        st.warning("Please upload a photo first.")
    else:
        with st.spinner(f"Generating {style_name} style... (~1-2 min on CPU, ~10s on GPU)"):
            output = run_style_transfer(
                content_img,
                style_name=style_name,
                strength=strength,
                guidance=guidance,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(content_img, caption="Original", use_container_width=True)
        with col_b:
            st.image(output, caption=f"✨ {style_name}", use_container_width=True)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        st.download_button("⬇️ Download", buf.getvalue(), "artwork.png", "image/png",
                           use_container_width=True)