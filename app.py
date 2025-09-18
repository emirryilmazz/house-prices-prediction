import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------- Config ----------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_pipeline(pipeline_path: Path):
    return joblib.load(pipeline_path)


def get_feature_columns():
    # Pipeline için gerekli sütunlar
    return [
        "bedrooms",
        "bathrooms", 
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "age",
        "sqft_per_floor",
        "city_grouped"
    ]


def make_input_dataframe(
    bedrooms: float,
    bathrooms: float,
    sqft_living: int,
    sqft_lot: int,
    floors: float,
    sqft_above: int,
    sqft_basement: int,
    age: int,
    city: str,
    view_score: int,
    condition_score: int,
    is_waterfront: bool,
):
    # Pipeline için basit format
    data = {
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": float(floors),
        "waterfront": 1 if is_waterfront else 0,  # bool -> int
        "view": int(view_score),
        "condition": int(condition_score),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "age": int(age),
        "sqft_per_floor": float(sqft_living) / float(floors) if float(floors) > 0 else float(sqft_living),
        "city_grouped": city
    }
    
    return pd.DataFrame([data])


def format_currency(value: float) -> str:
    try:
        return f"${value:,.0f}"
    except Exception:
        return str(value)


def main():
    st.title("🏠 House Price Predictor")
    st.caption("Basit ve modern bir arayüz ile modelinizi kullanın.")

    pipeline_path = Path("model/pipe.pkl")
    if not pipeline_path.exists():
        st.error("Pipeline dosyası bulunamadı: pipe.pkl")
        st.stop()

    with st.spinner("Pipeline yükleniyor..."):
        pipeline = load_pipeline(pipeline_path)

    st.subheader("Girdi Bilgileri")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            bedrooms = st.number_input("Yatak Odası (bedrooms)", min_value=0.0, max_value=10.0, value=3.0, step=1.0)
            bathrooms = st.number_input("Banyo (bathrooms)", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
            floors = st.number_input("Kat Sayısı (floors)", min_value=0.0, max_value=4.0, value=1.0, step=0.5)
            age = st.number_input("Yaş (age)", min_value=0, max_value=150, value=20, step=1)

        with c2:
            sqft_living = st.number_input("Yaşam Alanı (sqft_living)", min_value=100, max_value=20000, value=1800, step=50)
            sqft_lot = st.number_input("Arsa Alanı (sqft_lot)", min_value=200, max_value=100000, value=5000, step=50)
            sqft_above = st.number_input("Yer Üstü Alan (sqft_above)", min_value=0, max_value=20000, value=1600, step=50)
            sqft_basement = st.number_input("Bodrum Alanı (sqft_basement)", min_value=0, max_value=20000, value=200, step=50)

        # Derived feature display
        derived = 0 if floors == 0 else int(round(sqft_living / max(floors, 1), 0))
        st.info(f"Kat başına ortalama m² (sqft_per_floor): {derived}")

        city_options = [
            "Auburn",
            "Bellevue", 
            "Burien",
            "Federal Way",
            "Issaquah",
            "Kent",
            "Kirkland",
            "Maple Valley",
            "Mercer Island",
            "Other",
            "Redmond",
            "Renton",
            "Sammamish",
            "Seattle",
            "Shoreline",
            "Woodinville",
        ]
        c3, c4, c5 = st.columns(3)
        with c3:
            city = st.selectbox("Şehir (city)", options=city_options, index=12)  # Default Seattle
        with c4:
            view_score = st.select_slider("Manzara (view 0-4)", options=[0, 1, 2, 3, 4], value=0)
        with c5:
            condition_score = st.selectbox("Kondisyon (3-5)", options=[3, 4, 5], index=1)

        is_waterfront = st.toggle("Su Kenarı (waterfront)", value=False)

        submitted = st.form_submit_button("Fiyatı Tahmin Et")

    if submitted:
        try:
            X = make_input_dataframe(
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                sqft_living=int(sqft_living),
                sqft_lot=int(sqft_lot),
                floors=float(floors),
                sqft_above=int(sqft_above),
                sqft_basement=int(sqft_basement),
                age=int(age),
                city=city,
                view_score=int(view_score),
                condition_score=int(condition_score),
                is_waterfront=bool(is_waterfront),
            )

            
            # Pipeline ile tahmin yap
            pred = pipeline.predict(X)[0]

            st.success("Tahmin Başarılı")
            st.metric(label="Tahmini Fiyat", value=format_currency(pred))

            with st.expander("Girdi Özeti"):
                st.dataframe(X.T.rename(columns={0: "value"}))

        except Exception as e:
            st.error("Tahmin sırasında bir hata oluştu. Ayrıntı için aşağıya bakın.")
            st.exception(e)


if __name__ == "__main__":
    main()


