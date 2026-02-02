import os
from io import BytesIO

import requests
import streamlit as st


DEFAULT_API_URL = os.environ.get("POTATO_API_URL", "http://localhost:8080")


def _predict_via_api(api_base_url: str, filename: str, file_bytes: bytes) -> dict:
    api_base_url = api_base_url.rstrip("/")
    url = f"{api_base_url}/predict"

    files = {
        "file": (filename or "image.jpg", BytesIO(file_bytes), "application/octet-stream"),
    }

    response = requests.post(url, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="centered",
)

st.title("Potato Disease Classification")
st.caption("Drag & drop an image to predict: Early Blight / Late Blight / Healthy")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL, help="Example: http://localhost:8080")
    show_raw = st.checkbox("Show raw response", value=False)

st.markdown("---")

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    help="You can drag and drop an image here.",
)

if uploaded is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

image_bytes = uploaded.getvalue()

left, right = st.columns([1, 1])
with left:
    st.image(image_bytes, caption=uploaded.name, use_container_width=True)

with right:
    st.write(" ")
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Running prediction..."):
            try:
                result = _predict_via_api(api_url, uploaded.name, image_bytes)
            except requests.exceptions.RequestException as exc:
                st.error(
                    "Could not reach the API. Make sure it is running at the configured URL.\n\n"
                    f"Details: {exc}"
                )
                st.stop()

        predicted_class = result.get("class", "(missing 'class')")
        confidence = result.get("confidence", None)

        st.success("Prediction complete")

        st.metric("Predicted class", predicted_class)

        if isinstance(confidence, (int, float)):
            pct = max(0.0, min(1.0, float(confidence)))
            st.metric("Confidence", f"{pct * 100:.2f}%")
            st.progress(pct)
        else:
            st.warning("Response did not include numeric 'confidence'.")

        if show_raw:
            st.code(result, language="json")
