# BTK-LLMS-26 · Antalya LLM Tabanlı Uygulama Geliştirme Eğitimi

Büyük Dil Modelleri (LLM) ile pratik uygulamalar geliştirmeye yönelik eğitim materyali ve çalışan kod örnekleri.

**Yönetici:** Dr. Murat Altun | **Format:** Jupyter Notebook + Python  
**Ön Koşul:** Python 3.8+, GPU tercih edilir (Kaggle/Colab ile ücretsiz)

---

## 📚 Ana Projeler

### 1. **Gemma 3 Fine-Tuning — JSON Çıkarma** 
`gemma3-json-finetune-main/`

Fatura, tıbbi kayıt, sipariş gibi dağınık metinlerden geçerli JSON üretmek için küçük bir dil modelini optimize etme.

| Metrik | Sonuç |
|---|---|
| Geçerli JSON oranı | %0 → %95 |
| VRAM | 4.6 GB (yükleme) + 5.9 GB (eğitim) |
| Model | `unsloth/gemma-3-4b-it` (QLoRA, 4-bit) |
| Veri | `paraloq/json_data_extraction` (484 örnek) |
| Donanım | Ücretsiz Kaggle/Colab T4 GPU |

**Hızlı Başlangıç:**
1. [kaggle.com](https://www.kaggle.com) → hesap + telefon doğrula
2. Create → Notebook → Import → `gemma3_json_finetune.ipynb`
3. Accelerator = GPU T4 x2, Internet = On
4. Run All → ~15 dakika

👉 [NASIL-CALISTIRILIR.md](./gemma3-json-finetune-main/NASIL-CALISTIRILIR.md)

---

### 2. **MT5 Turkish Summarization — Metin Özetleme**
`mt5-turkish-summarization/`

Türkçe metinleri kısa ve anlamlı özetlere dönüştüren mT5 modelinin fine-tuning projesi.

| Bilgi | Detay |
|---|---|
| Model | mT5-small pretrained (101 dili destekler) |
| Mimari | Sequence-to-Sequence (metin → özet) |
| Veri | Türkçe metin-özet çiftleri |
| Kullanım | Haber özetleme, dokümantasyon, arama snippet'leri |

**Hızlı Kullanım:**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-small-turkish-summarization")
tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-small-turkish-summarization")

metin = "Türkiye ekonomisi son yıllarda önemli değişimler geçirmektedir..."
inputs = tokenizer.encode(metin, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=150, num_beams=4)
ozet = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

👉 [Model Hub](https://huggingface.co/ozcangundes/mt5-small-turkish-summarization) | [Colab](https://colab.research.google.com/drive/1zAsD0-0JQ6IQENudEog-bk8aBAGwAzHt?usp=sharing)

---

### 3. **Car Prediction** `car-predict/`
Araba fiyat tahmini projesi. Detaylı bilgi için dizin içindeki README'ye bakınız.

### 4. **Fine-tuning Laboratuvarı** `Finetune/`
Detaylı bilgi için dizin içindeki README'ye bakınız.

---

## 🤖 Hugging Face Projeleri

HF Hub'da paylaşılan, Colab üzerinde çalıştırılabilir pratik projeler.

### 📌 1. Pipeline ile Model Kullanımı

**Amaç:** Transformers kütüphanesinin Pipeline API'sı ile modelleri kolayca kullanmayı öğrenme.

**Görev Türleri:** Text classification, sentiment analysis, NER, question answering, vb.

**Başlangıç:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zAsD0-0JQ6IQENudEog-bk8aBAGwAzHt?usp=sharing)

**İçerik:** Transformers kütüphanesi, Pipeline API, Model seçimi, Batch işleme, Türkçe & İngilizce örnekler

---

### 📌 2. Hugging Face Model Kullanım Sunusu

Hugging Face Hub'daki modellerin yüklenmesi, fine-tuning'i ve inference'ı hakkında kapsamlı sunum.

**Kapsam:** Hub navigasyonu, AutoModel/AutoTokenizer, Tokenizasyon, Transfer learning, Fine-tuning, Quantization, Çok dilli modeller

**Hedef Kitle:** ML/NLP başlayanları, Production mühendisleri, Akademik araştırmacılar, Startup geliştiricileri

---

### 📌 3. Hugging Face API Inference — HF TOKEN ile Model Sorgulama

**Amaç:** Hugging Face Hub'daki modelleri doğrudan API üzerinden sorgulamayı ve Inference API'yi kullanarak çıkarımı öğrenme.

**Kapsam:** HF Inference API, API Token yönetimi, HTTP istekleri, Model API endpointleri, Eşzamanlı sorgular, Hata yönetimi

**İçerik:** HuggingFace API dokumentasyonu, Token oluşturma ve güvenli kullanım, Çeşitli görev türleri (classification, generation, Q&A, vb.), Türkçe & İngilizce örnekler, Rate limiting ve best practices

**Başlangıç:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xAzy3kpDNssph31jl8MpMoqPdZytOQ2g?usp=sharing)

**Temel Kullanım:**
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/MODEL_NAME"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

result = query({"inputs": "Merhaba, benim adım..."})
print(result)
```

---

## 📂 Tüm Projeler & Linkler

**Google Drive'da HF projeleri ve ek materyalleri görüntüleyin:**

👉 [Google Drive Klasörü](https://drive.google.com/drive/u/0/folders/1580A5Uj_tKj_88lEBJZIWYDMUaBmkYh2)

---

## 🛠 Teknik Stack

| Bileşen | Araç |
|---|---|
| **Dil** | Python 3.8+ |
| **Notebook** | Jupyter (`.ipynb`) |
| **Model Hub** | Hugging Face |
| **Çalıştırma** | Kaggle/Google Colab |
| **GPU** | T4 (ücretsiz) |
| **Kütüphaneler** | `torch`, `transformers`, `unsloth`, `peft`, `datasets`, `requests` |

---

## 🔗 Kaynaklar

- [Gemma Modelleri](https://huggingface.co/collections/google/gemma-release-65d746f89d895876a249ce11)
- [mT5 Dokümantasyonu](https://huggingface.co/docs/transformers/model_doc/mt5)
- [Hugging Face Hub](https://huggingface.co)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- [Unsloth Kütüphanesi](https://github.com/unslothai/unsloth)
- [LoRA Makalesi](https://arxiv.org/abs/2106.09685)
- [QLoRA Makalesi](https://arxiv.org/abs/2305.14314)

---

## 📝 Lisans

Eğitim materyali. Model ve veri seti kendi lisanslarına tabidir (Gemma / Apache 2.0).

---

## 🤝 Katkı & Soru

Sorular, hata bildirimleri veya iyileştirmeler için **Issues** açabilirsiniz.

---

**Eğitim Bilgisi**  
Antalya — Büyük Dil Modelleri (LLMs) Tabanlı Uygulama Geliştirme Eğitimi  
Dr. Murat Altun · 2024
