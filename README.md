# House Price Predictor

Basit ve modern bir Streamlit arayfczfc ile `model/model.pkl` i7indeki e1fitilmi5f modeli kullanarak fiyat tahmini yapar.

## Kurulum

1. Gerekli paketleri yfckleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamay31 e7al315ft31r31n:
```bash
streamlit run app.py
```

## Kullan31m
- Sayfadaki girdi alanlar31n31 doldurun (oda say31s31, banyo, m2 vb.).
- 5eehir, manzara (0-4), kondisyon (3-5) ve waterfront se317imlerini yap31n.
- "Fiyat31 Tahmin Et" butonuna bas31n.

Uygulama, modelin bekledi1fi f6zellik s31ras31n31 (one-hot encoded kolonlar dahil) otomatik olarak olu5fturur.
