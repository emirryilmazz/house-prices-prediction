import pickle
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
def load_model_and_scaler(model_path: Path):
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model


def get_feature_columns():
    # Order must match training feature order (df.info() minus target `price`)
    numeric_columns = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "age",
        "sqft_per_floor",
    ]

    city_columns = [
        "city_Auburn",
        "city_Bellevue",
        "city_Federal Way",
        "city_Issaquah",
        "city_Kenmore",
        "city_Kent",
        "city_Kirkland",
        "city_Maple Valley",
        "city_Other",
        "city_Redmond",
        "city_Renton",
        "city_Sammamish",
        "city_Seattle",
        "city_Shoreline",
        "city_Snoqualmie",
        "city_Woodinville",
    ]

    view_columns = [f"view_{i}" for i in range(5)]  # 0..4

    condition_columns = [
        "condition_3",
        "condition_4",
        "condition_5",
    ]

    waterfront_columns = [
        "waterfront_False",
        "waterfront_True",
    ]

    return numeric_columns + city_columns + view_columns + condition_columns + waterfront_columns


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
    columns = get_feature_columns()

    # Start with all zeros / False
    data = {col: 0 for col in columns}

    # Numeric features
    data["bedrooms"] = float(bedrooms)
    data["bathrooms"] = float(bathrooms)
    data["sqft_living"] = int(sqft_living)
    data["sqft_lot"] = int(sqft_lot)
    data["floors"] = float(floors)
    data["sqft_above"] = int(sqft_above)
    data["sqft_basement"] = int(sqft_basement)
    data["age"] = int(age)
    data["sqft_per_floor"] = (
        float(sqft_living) / float(floors) if float(floors) > 0 else float(sqft_living)
    )

    # One-hot: city
    city_key = f"city_{city}"
    if city_key in data:
        data[city_key] = True
    else:
        # Fallback to Other if unseen
        data["city_Other"] = True

    # One-hot: view 0..4
    view_key = f"view_{int(view_score)}"
    if view_key in data:
        data[view_key] = True

    # One-hot: condition 3/4/5
    cond_key = f"condition_{int(condition_score)}"
    if cond_key in data:
        data[cond_key] = True

    # One-hot: waterfront True/False
    data["waterfront_True"] = bool(is_waterfront)
    data["waterfront_False"] = not bool(is_waterfront)

    df = pd.DataFrame([data], columns=columns)
    # Ensure bool columns are bool dtype, numeric remain numeric
    bool_cols = [c for c in df.columns if c.startswith("city_") or c.startswith("view_") or c.startswith("condition_") or c.startswith("waterfront_")]
    df[bool_cols] = df[bool_cols].astype(bool)
    return df


def format_currency(value: float) -> str:
    try:
        return f"${value:,.0f}"
    except Exception:
        return str(value)


def main():
    st.title("🏠 House Price Predictor")
    st.caption("Basit ve modern bir arayüz ile modelinizi kullanın.")

    model_path = Path("model") / "model.pkl"
    if not model_path.exists():
        st.error("Model dosyası bulunamadı: model/model.pkl")
        st.stop()

    with st.spinner("Model yükleniyor..."):
        model = load_model_and_scaler(model_path)

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
            "Federal Way",
            "Issaquah",
            "Kenmore",
            "Kent",
            "Kirkland",
            "Maple Valley",
            "Other",
            "Redmond",
            "Renton",
            "Sammamish",
            "Seattle",
            "Shoreline",
            "Snoqualmie",
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

            
            # Model tahmin yap (normal scale'de)
            y_pred = model.predict(X)
            pred = float(np.array(y_pred).ravel()[0])

            st.success("Tahmin Başarılı")
            st.metric(label="Tahmini Fiyat", value=format_currency(pred))

            with st.expander("Girdi Özeti"):
                st.dataframe(X.T.rename(columns={0: "value"}))

        except Exception as e:
            st.error("Tahmin sırasında bir hata oluştu. Ayrıntı için aşağıya bakın.")
            st.exception(e)


if __name__ == "__main__":
    main()


