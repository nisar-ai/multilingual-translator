import html
import re
from typing import Dict, List, Tuple

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# =================================================
# PAGE SETTINGS
# =================================================
st.set_page_config(
    page_title="Nisar's Multilingual Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =================================================
# UI STYLING
# =================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0f172a;
        --blue: #2563eb;
        --cyan: #0891b2;
        --muted: #64748b;
        --border: #bfdbfe;
    }

    * {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        min-height: 100vh;
        background:
            radial-gradient(circle at top left, rgba(6, 182, 212, 0.26), transparent 32%),
            radial-gradient(circle at bottom right, rgba(99, 102, 241, 0.28), transparent 34%),
            linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #0e7490 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .app-shell {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.60);
        border-radius: 28px;
        padding: 1.4rem;
        box-shadow: 0 24px 70px rgba(2, 6, 23, 0.33);
        backdrop-filter: blur(18px);
    }

    .hero {
        background:
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.22), transparent 32%),
            linear-gradient(135deg, #1d4ed8, #0891b2);
        border-radius: 20px;
        padding: 1.75rem 1.4rem;
        margin-bottom: 1.25rem;
        color: white;
        text-align: center;
        box-shadow: 0 12px 28px rgba(30, 64, 175, 0.25);
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(1.8rem, 4vw, 2.7rem);
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0.65rem 0 0;
        color: rgba(255, 255, 255, 0.93);
        font-size: 1rem;
        font-weight: 500;
    }

    .content-card {
        height: 100%;
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .section-title {
        color: var(--navy);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .small-note {
        margin: 0.4rem 0 0.9rem;
        color: var(--muted);
        font-size: 0.85rem;
        line-height: 1.55;
    }

    .language-chip {
        min-height: 64px;
        padding: 0.65rem 0.35rem;
        background: linear-gradient(135deg, #ecfeff, #eff6ff);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        color: #0f172a;
        font-size: 0.88rem;
        font-weight: 700;
        text-align: center;
    }

    .result-card {
        margin-top: 0.85rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f8fafc, #eff6ff);
        border: 1px solid #bfdbfe;
        border-left: 5px solid #0891b2;
        border-radius: 14px;
    }

    .result-heading {
        margin-bottom: 0.55rem;
        color: #075985;
        font-size: 0.94rem;
        font-weight: 800;
    }

    .result-text {
        padding: 0.85rem 0.95rem;
        background: #ffffff;
        border-radius: 10px;
        color: #0f172a;
        font-size: 1rem;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }

    .about-box {
        margin-top: 1.25rem;
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(135deg, #312e81, #7c3aed, #db2777);
        box-shadow: 0 14px 35px rgba(109, 40, 217, 0.28);
    }

    .about-box h2 {
        margin: 0 0 0.5rem;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .about-box p {
        margin: 0.35rem 0;
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .footer {
        margin-top: 1rem;
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.82rem;
        text-align: center;
    }

    .warning-note {
        margin-top: 0.85rem;
        padding: 0.8rem 0.9rem;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 11px;
        color: #9a3412;
        font-size: 0.83rem;
        line-height: 1.5;
    }

    .stSelectbox > label,
    .stTextArea > label {
        display: none !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        min-height: 46px !important;
        background: #f8fafc !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 12px !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea {
        min-height: 190px !important;
        padding: 1rem !important;
        background: #f8fafc !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 14px !important;
        color: #0f172a !important;
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }

    .stTextArea textarea:focus {
        border-color: #0891b2 !important;
        box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.15) !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border: none !important;
        border-radius: 13px !important;
        background: linear-gradient(135deg, #2563eb, #0891b2) !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
    }

    .stButton > button:hover {
        border: none !important;
        background: linear-gradient(135deg, #1d4ed8, #0e7490) !important;
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.35);
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 1rem 0.7rem 2rem;
        }

        .app-shell {
            padding: 0.85rem;
            border-radius: 20px;
        }

        .hero {
            padding: 1.35rem 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =================================================
# LANGUAGE SETTINGS FOR NLLB-200
# =================================================
LANGUAGES = {
    "English": {
        "code": "eng_Latn",
        "flag": "🇺🇸",
    },
    "Urdu": {
        "code": "urd_Arab",
        "flag": "🇵🇰",
    },
    "Arabic": {
        "code": "arb_Arab",
        "flag": "🇸🇦",
    },
    "German": {
        "code": "deu_Latn",
        "flag": "🇩🇪",
    },
}


# =================================================
# NAME GLOSSARY
#
# Add or edit preferred spellings here.
# =================================================
NAME_TRANSLATIONS = {
    "Nisar Ahmad": {
        "English": "Nisar Ahmad",
        "Urdu": "نثار احمد",
        "Arabic": "نزار أحمد",
        "German": "Nisar Ahmad",
    },
    "Nisar": {
        "English": "Nisar",
        "Urdu": "نثار",
        "Arabic": "نزار",
        "German": "Nisar",
    },
    "Ahmad": {
        "English": "Ahmad",
        "Urdu": "احمد",
        "Arabic": "أحمد",
        "German": "Ahmad",
    },
}


# =================================================
# MODEL LOADING
# =================================================
@st.cache_resource(show_spinner="Loading translation model for the first time...")
def load_model():
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model.to(device)
        model.eval()

        return model, tokenizer, device, None

    except Exception as error:
        return None, None, None, str(error)


model, tokenizer, device, model_error = load_model()


# =================================================
# PROTECTION PATTERNS
#
# These preserve items that must not be translated:
# emails, URLs, filenames, values, times, and dates.
# =================================================
PROTECTED_PATTERNS = [
    r"\b[\w.\-+]+@[\w.\-]+\.\w+\b",                     # Email
    r"https?://[^\s]+",                                # URL
    r"\b[\w\-]+\.(?:pdf|docx|doc|xlsx|xls|pptx|txt|csv|py|json)\b",
    r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\b",            # Time
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",               # 150,000 / 1,500.50
    r"\b\d+(?:\.\d+)?\s?(?:°C|°F|km|kilometers?|kg|kilograms?|MB|GB)\b",
]


# =================================================
# TEXT HELPER FUNCTIONS
# =================================================
def normalise_spaces(text: str) -> str:
    """Remove accidental repeated spaces."""
    return re.sub(r"\s+", " ", text).strip()


def protect_special_content(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace protected text with simple placeholders before translation.
    These placeholders are restored after translation.
    """
    replacements = {}
    protected_text = text
    item_number = 0

    for pattern in PROTECTED_PATTERNS:
        matches = list(re.finditer(pattern, protected_text, flags=re.IGNORECASE))

        # Replace from last match to first, so positions remain valid.
        for match in reversed(matches):
            original_value = match.group(0)
            placeholder = f"XPROTECTEDITEM{item_number}X"

            replacements[placeholder] = original_value

            protected_text = (
                protected_text[:match.start()]
                + placeholder
                + protected_text[match.end():]
            )

            item_number += 1

    return protected_text, replacements


def restore_special_content(text: str, replacements: Dict[str, str]) -> str:
    """Restore protected text after translation."""
    restored_text = text

    for placeholder, original_value in replacements.items():
        variants = [
            placeholder,
            placeholder.lower(),
            placeholder.upper(),
            placeholder.replace("X", " X "),
            placeholder.replace("X", ""),
        ]

        for variant in variants:
            restored_text = restored_text.replace(variant, original_value)

    return restored_text


def source_contains_name(source_text: str, name: str) -> bool:
    """Case-insensitive name check for source text."""
    return bool(
        re.search(
            re.escape(name),
            source_text,
            flags=re.IGNORECASE,
        )
    )


def correct_known_names(
    source_text: str,
    translated_text: str,
    target_language: str,
) -> str:
    """
    Apply a small name glossary after model output.
    This is useful because translation models can mistranslate names.
    """
    corrected_text = translated_text

    # First correct the full name.
    if source_contains_name(source_text, "Nisar Ahmad"):
        desired_name = NAME_TRANSLATIONS["Nisar Ahmad"][target_language]

        common_variants = [
            "Nisar Ahmad",
            "Nisar ahmad",
            "nisar ahmad",
            "نثار احمد",
            "نثار أحمد",
            "نظام احمد",
            "نظام أحمد",
            "نزار احمد",
            "نزار أحمد",
            "نيزار أحمد",
            "نيزار احمد",
        ]

        for variant in common_variants:
            corrected_text = re.sub(
                re.escape(variant),
                desired_name,
                corrected_text,
                flags=re.IGNORECASE,
            )

    # Then correct the first name when it appears without Ahmad.
    elif source_contains_name(source_text, "Nisar"):
        desired_name = NAME_TRANSLATIONS["Nisar"][target_language]

        if target_language == "Urdu":
            corrected_text = re.sub(
                r"\bنظام\b",
                desired_name,
                corrected_text,
            )

        if target_language == "Arabic":
            corrected_text = re.sub(
                r"\bنيزار\b",
                desired_name,
                corrected_text,
            )

    return normalise_spaces(corrected_text)


# =================================================
# TRANSLATION FUNCTION
# =================================================
def translate_text(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Translate text with NLLB-200 and restore important literal values.
    """
    source_code = LANGUAGES[source_language]["code"]
    target_code = LANGUAGES[target_language]["code"]

    # Protect email addresses, files, numeric values, etc.
    protected_text, replacements = protect_special_content(text)

    tokenizer.src_lang = source_code

    encoded = tokenizer(
        protected_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    target_token_id = tokenizer.convert_tokens_to_ids(target_code)

    with torch.no_grad():
        output_tokens = model.generate(
            **encoded,
            forced_bos_token_id=target_token_id,
            max_new_tokens=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    translated_text = tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True,
    )[0].strip()

    translated_text = restore_special_content(
        translated_text,
        replacements,
    )

    translated_text = correct_known_names(
        source_text=text,
        translated_text=translated_text,
        target_language=target_language,
    )

    return translated_text


# =================================================
# SESSION STATE
# =================================================
if "translations" not in st.session_state:
    st.session_state.translations = {}

if "last_source_language" not in st.session_state:
    st.session_state.last_source_language = ""

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""


# =================================================
# APP INTERFACE
# =================================================
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🌍 Multilingual Translator</h1>
        <p>English • Urdu • Arabic • German</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([0.85, 1.45], gap="large")


# =================================================
# LEFT PANEL
# =================================================
with left_column:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🎯 Source language</div>',
        unsafe_allow_html=True,
    )

    source_language = st.selectbox(
        "Source language",
        options=list(LANGUAGES.keys()),
        format_func=lambda language: (
            f"{LANGUAGES[language]['flag']} {language}"
        ),
        label_visibility="collapsed",
    )

    target_languages = [
        language
        for language in LANGUAGES.keys()
        if language != source_language
    ]

    st.markdown(
        """
        <div class="section-title" style="margin-top:1.25rem;">
            🔄 Translation targets
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="small-note">
            Your text will be translated into all three remaining languages.
        </p>
        """,
        unsafe_allow_html=True,
    )

    target_columns = st.columns(3)

    for column, language in zip(target_columns, target_languages):
        with column:
            st.markdown(
                f"""
                <div class="language-chip">
                    {LANGUAGES[language]["flag"]}<br>
                    {language}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if model_error:
        st.error("❌ Translation model could not be loaded.")

        with st.expander("View model error"):
            st.code(model_error)

    else:
        device_label = "GPU" if device == "cuda" else "CPU"

        st.markdown(
            f"""
            <p class="small-note" style="margin-top:1.2rem;">
                ⚙️ NLLB-200 model ready • Running on {device_label}
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# RIGHT PANEL
# =================================================
with right_column:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">✍️ Enter your text</div>',
        unsafe_allow_html=True,
    )

    input_text = st.text_area(
        "Text to translate",
        placeholder=(
            "Write or paste your text here...\n\n"
            "Examples:\n"
            "Hello, my name is Nisar Ahmad.\n"
            "السلام علیکم، میرا نام نثار احمد ہے۔\n"
            "Hallo, wie geht es dir?"
        ),
        height=195,
        label_visibility="collapsed",
    )

    translate_clicked = st.button(
        "🚀 Translate into 3 languages",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="warning-note">
            ⚠️ AI-assisted translation: please review important names,
            dates, numbers, legal text, medical information, financial
            details, and technical content before use.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if translate_clicked:
        if not input_text.strip():
            st.warning("⚠️ Please enter text before translating.")

        elif model is None:
            st.error("❌ The translation model is unavailable.")

        else:
            st.session_state.translations = {}

            try:
                with st.spinner("Creating translations..."):
                    for target_language in target_languages:
                        st.session_state.translations[target_language] = (
                            translate_text(
                                text=input_text.strip(),
                                source_language=source_language,
                                target_language=target_language,
                            )
                        )

                st.session_state.last_source_language = source_language
                st.session_state.last_input_text = input_text.strip()

                st.success(
                    f"✅ Translated from {source_language} into "
                    f"{', '.join(target_languages)}."
                )

            except Exception as error:
                st.error(f"❌ Translation failed: {str(error)[:180]}")

    if st.session_state.translations:
        st.markdown(
            """
            <div class="section-title" style="margin-top:1.35rem;">
                ✨ Translation results
            </div>
            """,
            unsafe_allow_html=True,
        )

        for language, translated_text in st.session_state.translations.items():
            flag = LANGUAGES[language]["flag"]

            # Escape output so translated content cannot break HTML/CSS.
            safe_text = html.escape(translated_text).replace("\n", "<br>")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-heading">{flag} {language}</div>
                    <div class="result-text">{safe_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# YOUR INTRODUCTION
# =================================================
st.markdown(
    """
    <div class="about-box">
        <h2>👨‍💻 About the Developer</h2>
        <p><strong>Nisar Ahmad</strong> — AI Developer</p>
        <p>
            I am a student at COMSATS University Islamabad, Sahiwal Campus,
            interested in Artificial Intelligence, machine learning, and
            creating useful AI-powered applications.
        </p>
        <p>
            This project is a multilingual translation tool that supports
            English, Urdu, Arabic, and German in one simple interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
        🌟 Built with Streamlit, PyTorch, and NLLB-200 • By Nisar Ahmad
    </div>
    """,
    unsafe_allow_html=True,
)
