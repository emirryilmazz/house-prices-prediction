import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_model():
    """Modeli yükle"""
    model_path = Path("model") / "model.pkl"
    with model_path.open("rb") as f:
        return pickle.load(f)

def create_sample_data():
    """Örnek veri oluştur - modelinizin beklediği formatta"""
    # Numeric features
    data = {
        'bedrooms': 3.0,
        'bathrooms': 2.0,
        'sqft_living': 1800,
        'sqft_lot': 5000,
        'floors': 1.0,
        'sqft_above': 1600,
        'sqft_basement': 200,
        'age': 20,
        'sqft_per_floor': 1800.0,  # sqft_living / floors
    }
    
    # City one-hot encoding (Seattle seçili)
    cities = [
        'city_Auburn', 'city_Bellevue', 'city_Federal Way', 'city_Issaquah',
        'city_Kenmore', 'city_Kent', 'city_Kirkland', 'city_Maple Valley',
        'city_Other', 'city_Redmond', 'city_Renton', 'city_Sammamish',
        'city_Seattle', 'city_Shoreline', 'city_Snoqualmie', 'city_Woodinville'
    ]
    for city in cities:
        data[city] = city == 'city_Seattle'  # Seattle seçili
    
    # View one-hot encoding (view_0 seçili)
    for i in range(5):
        data[f'view_{i}'] = i == 0  # view_0 seçili
    
    # Condition one-hot encoding (condition_4 seçili)
    for i in [3, 4, 5]:
        data[f'condition_{i}'] = i == 4  # condition_4 seçili
    
    # Waterfront one-hot encoding (False seçili)
    data['waterfront_False'] = True
    data['waterfront_True'] = False
    
    return pd.DataFrame([data])

def main():
    print("Model yükleniyor...")
    try:
        model = load_model()
        print("✅ Model başarıyla yüklendi")
        
        print("\nÖrnek veri oluşturuluyor...")
        X = create_sample_data()
        print(f"✅ Veri boyutu: {X.shape}")
        
        print("\nTahmin yapılıyor...")
        prediction = model.predict(X)
        predicted_price = float(np.array(prediction).ravel()[0])
        
        print(f"🏠 Tahmini fiyat: ${predicted_price:,.0f}")
        
        print("\nKullanılan özellikler:")
        print("- Yatak odası: 3")
        print("- Banyo: 2")
        print("- Yaşam alanı: 1,800 sqft")
        print("- Arsa alanı: 5,000 sqft")
        print("- Kat sayısı: 1")
        print("- Yaş: 20")
        print("- Şehir: Seattle")
        print("- Manzara: 0")
        print("- Kondisyon: 4")
        print("- Su kenarı: Hayır")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
