import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="Skin Lesion Analyser",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #161b27; }
  h1 { font-size: 2rem !important; }
  .metric-box {
      background: #1e2535;
      border: 1px solid #2e3a52;
      border-radius: 10px;
      padding: 1rem 1.2rem;
      text-align: center;
  }
  .metric-label { color: #8892a4; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric-value { color: #e8eaf0; font-size: 1.5rem; font-weight: 600; margin-top: 4px; }
  .verdict-malignant {
      background: #3a1a1a; border: 1px solid #8b3030;
      color: #f87171; border-radius: 8px;
      padding: 1rem 1.5rem; text-align: center; font-size: 1.2rem; font-weight: 600;
  }
  .verdict-benign {
      background: #0f2a20; border: 1px solid #1e6b47;
      color: #34d399; border-radius: 8px;
      padding: 1rem 1.5rem; text-align: center; font-size: 1.2rem; font-weight: 600;
  }
  .feat-label { color: #8892a4; font-size: 0.78rem; font-family: monospace; }
  .pipeline-step {
      background: #1e2535; border: 1px solid #2e3a52;
      border-radius: 8px; padding: 0.5rem 0.8rem;
      font-size: 0.78rem; color: #8892a4; text-align: center;
  }
  .pipeline-step.done { border-color: #1e6b47; color: #34d399; background: #0f2a20; }
  .pipeline-step.active { border-color: #3b5bdb; color: #748ffc; background: #1a1f3a; }
  .model-badge {
      display: inline-block;
      padding: 2px 8px; border-radius: 4px;
      font-size: 0.72rem; font-weight: 600; margin-left: 6px;
  }
  .badge-fusion { background: #1a1f3a; color: #748ffc; border: 1px solid #3b5bdb; }
  .badge-image  { background: #0f2a20; color: #34d399; border: 1px solid #1e6b47; }
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_REGISTRY: list[tuple[str, str, str]] = [

    (
        "⭐ ResNet18 + Skin Fusion  [Best]",
        os.path.join(BASE_DIR, "Best feature Fusion Paths", "best_resnet_model.pth"),
        "fusion_resnet",
    ),
    (
        "⭐ DenseNet121 + Skin Fusion  [Best]",
        os.path.join(BASE_DIR, "Best feature Fusion Paths", "Densenet.pth"),
        "fusion_densenet",
    ),
    (
        "⭐ InceptionV3 + Skin Fusion  [Best]",
        os.path.join(BASE_DIR, "Best feature Fusion Paths", "best_inception_model.pth"),
        "fusion_inception",
    ),
    (
        "⭐ Ensemble + Skin Fusion  [Best]",
        os.path.join(BASE_DIR, "Best feature Fusion Paths", "Ensemble_with.pth"),
        "fusion_resnet",          # ensemble saved as resnet-based fusion head
    ),
    (
        "ResNet18  [Image-only]",
        os.path.join(BASE_DIR, "Normal Fusion Paths", "normal_resnet.pth"),
        "resnet",
    ),
    (
        "DenseNet121  [Image-only]",
        os.path.join(BASE_DIR, "Normal Fusion Paths", "normal_densenet.pth"),
        "densenet",
    ),
    (
        "InceptionV3  [Image-only]",
        os.path.join(BASE_DIR, "Normal Fusion Paths", "normal_inception.pth"),
        "inception",
    ),
]


AVAILABLE_MODELS = [m for m in MODEL_REGISTRY if os.path.exists(m[1])]
MODEL_LABELS = [m[0] for m in AVAILABLE_MODELS]


NUM_FEATURES = 15   


class FusionResNet(nn.Module):
    """ResNet-18 backbone fused with skin-tone tabular features."""
    def __init__(self, num_skin_features: int):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, image, skin_feat):
        img_feat = self.cnn(image)
        skin_feat = self.skin_mlp(skin_feat)
        fused = torch.cat([img_feat, skin_feat], dim=1)
        return self.classifier(fused)


class FusionDenseNet(nn.Module):
    """DenseNet-121 backbone fused with skin-tone tabular features."""
    def __init__(self, num_skin_features: int):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, image, skin_feat):
        feat = self.features(image)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        img_feat = feat.view(feat.size(0), -1)
        skin_feat = self.skin_mlp(skin_feat)
        fused = torch.cat([img_feat, skin_feat], dim=1)
        return self.classifier(fused)


class FusionInception(nn.Module):
    """InceptionV3 backbone fused with skin-tone tabular features."""
    def __init__(self, num_skin_features: int):
        super().__init__()
        base = models.inception_v3(weights=None, aux_logits=False)
        base.fc = nn.Identity()
        self.cnn = base
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, image, skin_feat):
        img_feat = self.cnn(image)
        skin_feat = self.skin_mlp(skin_feat)
        fused = torch.cat([img_feat, skin_feat], dim=1)
        return self.classifier(fused)


class ImageOnlyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.model = base

    def forward(self, image):
        return self.model(image)


class ImageOnlyDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        base.classifier = nn.Linear(base.classifier.in_features, 1)
        self.model = base

    def forward(self, image):
        return self.model(image)


class ImageOnlyInception(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.inception_v3(weights=None, aux_logits=False)
        base.fc = nn.Linear(2048, 1)
        self.model = base

    def forward(self, image):
        return self.model(image)


def is_fusion_type(model_type: str) -> bool:
    return model_type.startswith("fusion")


def build_architecture(model_type: str) -> nn.Module:
    if model_type == "fusion_resnet":
        return FusionResNet(num_skin_features=NUM_FEATURES)
    elif model_type == "fusion_densenet":
        return FusionDenseNet(num_skin_features=NUM_FEATURES)
    elif model_type == "fusion_inception":
        return FusionInception(num_skin_features=NUM_FEATURES)
    elif model_type == "resnet":
        return ImageOnlyResNet()
    elif model_type == "densenet":
        return ImageOnlyDenseNet()
    elif model_type == "inception":
        return ImageOnlyInception()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def extract_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([50, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def extract_features(pil_image: Image.Image) -> dict:
    """Extract the same 15 skin-tone features as feature_extraction.py."""
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (512, 512))

    skin_mask = extract_skin_mask(img_bgr)
    skin_pixels = img_bgr[skin_mask > 0]

    if len(skin_pixels) == 0:
        skin_pixels = img_bgr.reshape(-1, 3)

    avg_color = np.mean(skin_pixels, axis=0)
    med_color = np.median(skin_pixels, axis=0)
    std_color = np.std(skin_pixels, axis=0)

    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab_img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv_skin = hsv_img[skin_mask > 0] if skin_mask.any() else hsv_img.reshape(-1, 3)
    lab_skin = lab_img[skin_mask > 0]  if skin_mask.any() else lab_img.reshape(-1, 3)

    brightness_mean  = float(np.mean(hsv_skin[:, 2]))
    hue_mean         = float(np.mean(hsv_skin[:, 0]))
    saturation_mean  = float(np.mean(hsv_skin[:, 1]))
    lightness_mean   = float(np.mean(lab_skin[:, 0]))
    skin_ratio       = float(np.sum(skin_mask > 0) / (512 * 512))
    edges            = cv2.Canny(img_bgr, 100, 200)
    edge_density     = float(np.sum(edges > 0) / edges.size)

    return {
        "avg_R":           float(avg_color[2]),
        "avg_G":           float(avg_color[1]),
        "avg_B":           float(avg_color[0]),
        "med_R":           float(med_color[2]),
        "med_G":           float(med_color[1]),
        "med_B":           float(med_color[0]),
        "std_R":           float(std_color[2]),
        "std_G":           float(std_color[1]),
        "std_B":           float(std_color[0]),
        "brightness_mean": brightness_mean,
        "hue_mean":        hue_mean,
        "saturation_mean": saturation_mean,
        "lightness_mean":  lightness_mean,
        "skin_pixel_ratio":skin_ratio,
        "edge_density":    edge_density,
    }



IMG_TRANSFORM_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMG_TRANSFORM_299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_transform(model_type: str):
    return IMG_TRANSFORM_299 if "inception" in model_type else IMG_TRANSFORM_224


@st.cache_resource
def load_model(weights_path: str, model_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_architecture(model_type).to(device)
    weights_ok = False
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            net.load_state_dict(state, strict=False)
            weights_ok = True
        except Exception as e:
            st.warning(f"⚠️ Could not load weights: {e}")
    net.eval()
    return net, device, weights_ok

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    if not AVAILABLE_MODELS:
        st.error("No `.pth` model files found in expected folders.")
        selected_idx = None
    else:
        selected_label = st.selectbox(
            "🧠 Select model",
            options=MODEL_LABELS,
            index=0,
            help="Choose a pre-trained model checkpoint to run inference with.",
        )
        selected_idx = MODEL_LABELS.index(selected_label)
        _, sel_path, sel_type = AVAILABLE_MODELS[selected_idx]

        kind = "Skin-Fusion" if is_fusion_type(sel_type) else "Image-Only"
        badge_cls = "badge-fusion" if is_fusion_type(sel_type) else "badge-image"
        st.markdown(
            f'<span class="model-badge {badge_cls}">{kind}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"`{os.path.basename(sel_path)}`")

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("""
- **Fusion models** use ResNet-18 / DenseNet-121 / InceptionV3 backbone + skin-tone MLP branch
- **Image-only models** use the same backbones with a single linear head
- **Loss:** BCEWithLogitsLoss
- **Dataset:** ISIC 2018
    """)
    st.markdown("---")
    st.caption("⚠️ Research demo only. Not a medical device.")


st.markdown("# 🔬 Skin Lesion Malignancy Detector")
st.markdown("Upload a dermoscopy image. Select a model from the sidebar and run inference.")


cols_pipe = st.columns(5)
pipe_labels = ["1 · Upload", "2 · Extract features", "3 · CNN encode", "4 · Fusion MLP", "5 · Predict"]
pipe_placeholders = [c.empty() for c in cols_pipe]

def draw_pipeline(active_step: int):
    for i, (ph, label) in enumerate(zip(pipe_placeholders, pipe_labels)):
        if i < active_step:
            cls = "done"
        elif i == active_step:
            cls = "active"
        else:
            cls = ""
        ph.markdown(f'<div class="pipeline-step {cls}">{label}</div>', unsafe_allow_html=True)

draw_pipeline(-1)

st.markdown("---")

uploaded = st.file_uploader(
    "Drop a dermoscopy image (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded and selected_idx is not None:
    draw_pipeline(0)
    pil_img = Image.open(uploaded).convert("RGB")

    _, sel_path, sel_type = AVAILABLE_MODELS[selected_idx]
    fusion = is_fusion_type(sel_type)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Input image")
        st.image(pil_img, use_container_width=True)

        if fusion:
            st.markdown("#### Extracted skin features")
            draw_pipeline(1)
            with st.spinner("Extracting features…"):
                feats = extract_features(pil_img)

            feat_names = list(feats.keys())
            feat_vals  = list(feats.values())

            maxv = max(abs(v) for v in feat_vals) or 1.0
            for name, val in feats.items():
                norm = abs(val) / maxv
                bar  = "█" * int(norm * 20) + "░" * (20 - int(norm * 20))
                st.markdown(
                    f'<span class="feat-label">{name:<20} {bar}  {val:8.3f}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("ℹ️ Image-only model — skin feature extraction skipped.")
            feat_names, feat_vals = [], []

    with right:
        st.markdown("#### Prediction")

        model, device, weights_loaded = load_model(sel_path, sel_type)

        if not weights_loaded:
            st.warning("⚠️ Weights could not be loaded — running with **random weights**.")
        else:
            st.success(f"✅ Loaded `{os.path.basename(sel_path)}`")

        if st.button("▶  Run inference", use_container_width=True, type="primary"):
            with st.spinner("CNN encoding…"):
                draw_pipeline(2)
                transform = get_transform(sel_type)
                img_tensor = transform(pil_img).unsqueeze(0).to(device)

            draw_pipeline(3)

            with st.spinner("Running model…"):
                draw_pipeline(4)
                with torch.no_grad():
                    if fusion:
                        feat_tensor = torch.tensor(
                            [feat_vals], dtype=torch.float32
                        ).to(device)
                        logit = model(img_tensor, feat_tensor)
                    else:
                        logit = model(img_tensor)

                    prob = torch.sigmoid(logit).item()

            is_malignant = prob >= threshold
            verdict      = "Malignant" if is_malignant else "Benign"
            verdict_cls  = "verdict-malignant" if is_malignant else "verdict-benign"
            icon         = "⚠️" if is_malignant else "✅"

            st.markdown(
                f'<div class="{verdict_cls}">{icon} &nbsp; {verdict}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**Malignancy probability**")
            st.progress(prob)
            st.markdown(f"**{prob:.4f}**  *(threshold: {threshold:.2f})*")

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Probability</div><div class="metric-value">{prob:.3f}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Verdict</div><div class="metric-value">{verdict}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Threshold</div><div class="metric-value">{threshold:.2f}</div></div>', unsafe_allow_html=True)

            if fusion and feat_names:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 Feature values sent to model"):
                    import pandas as pd
                    df = pd.DataFrame({
                        "Feature": feat_names,
                        "Value":   [f"{v:.4f}" for v in feat_vals],
                    })
                    st.dataframe(df, hide_index=True, use_container_width=True)

elif uploaded and selected_idx is None:
    st.error("No models found. Please ensure `.pth` files exist in 'Best feature Fusion Paths' or 'Normal Fusion Paths'.")

else:
    st.markdown("""
    <div style="border:1.5px dashed #2e3a52;border-radius:12px;padding:3rem;text-align:center;color:#8892a4;margin-top:1rem;">
        📁 &nbsp; Upload a dermoscopy image to begin
    </div>
    """, unsafe_allow_html=True)