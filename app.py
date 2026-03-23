import streamlit as st
from PIL import Image
from style_transfer import run_style_transfer
import io

st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("🎨 Neural Style Transfer")
st.markdown("Transform your photo into a work of art using AI.")

col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("📷 Upload Content Image", type=["jpg", "jpeg", "png"])
    if content_file:
        content_img = Image.open(content_file).convert("RGB")
        st.image(content_img, caption="Content Image", use_column_width=True)

with col2:
    style_file = st.file_uploader("🖼️ Upload Style Image (e.g. Van Gogh, Monet)", type=["jpg", "jpeg", "png"])
    if style_file:
        style_img = Image.open(style_file).convert("RGB")
        st.image(style_img, caption="Style Image", use_column_width=True)

st.divider()

num_steps = st.slider("Optimization Steps (more = better quality, slower)", 100, 500, 300, 50)
style_weight = st.select_slider("Style Strength", options=[1e4, 1e5, 1e6, 1e7], value=1e6)

if st.button("✨ Generate Stylized Image"):
    if not content_file or not style_file:
        st.warning("Please upload both a content image and a style image.")
    else:
        with st.spinner("Running Neural Style Transfer... This may take 1-3 minutes on CPU."):
            output = run_style_transfer(
                content_img, style_img,
                num_steps=num_steps,
                style_weight=style_weight
            )

        st.success("Done!")
        st.image(output, caption="🎨 Stylized Output", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Stylized Image",
            data=buf.getvalue(),
            file_name="stylized_output.png",
            mime="image/png"
        )