# BTK-LLMS-26 · Antalya LLM Tabanlı Uygulama Geliştirme Eğitimi

Büyük Dil Modelleri (LLM) ile pratik uygulamalar geliştirmeye yönelik kapsamlı eğitim materyali ve çalışan kod örnekleri.

**Yönetici:** Dr. Murat Altun · **Format:** Jupyter Notebook + Python  
**Ön Koşul:** Python 3.8+, GPU tercih edilir (Kaggle/Colab ile ücretsiz)

---

## 📚 Projeler

### 1. **Gemma 3 Fine-Tuning — JSON Çıkarma** `gemma3-json-finetune-main/`

Dağınık metinlerden (fatura, tıbbi kayıt, sipariş…) **geçerli JSON üretmek** için küçük bir dil modelini optimize etme.

**Temel Sonuç:**
- ✅ Fine-tune **öncesi:** %0 geçerli JSON  
- ✅ Fine-tune **sonrası:** %95 geçerli JSON  
- ✅ **VRAM:** 4.6 GB yükleme + 5.9 GB eğitim (ücretsiz T4'e sığar)

**Teknoloji:**
- Model: `unsloth/gemma-3-4b-it` (Google açık ağırlık)
- Yöntem: **QLoRA** (4-bit) — sadece %0.5 parametre eğitimi
- Veri: `paraloq/json_data_extraction` (484 örnek, apache-2.0 lisansı)
- Donanım: Ücretsiz Kaggle/Colab **T4 GPU**

**İçerik:**
| Dosya | Açıklama |
|---|---|
| `gemma3_json_finetune.ipynb` | Ana notebook — teori + kod + baseline/sonrası ölçümü |
| `NASIL-CALISTIRILIR.md` | Adım adım çalıştırma rehberi (Kaggle + Colab) |
| `gemma3-json-finetune-sunu.pptx` | 11 slaytlık eğitim sunusu (konuşmacı notları) |

**Hızlı Başlangıç:**
```bash
1. kaggle.com → hesap + telefon doğrulama
2. Create → Notebook → Import → gemma3_json_finetune.ipynb
3. Sağ panel: Accelerator = GPU T4 x2, Internet = On
4. Run All → ~15 dakika → sonuç ekrana düşer
```
👉 Ayrıntı: [`gemma3-json-finetune-main/NASIL-CALISTIRILIR.md`](./gemma3-json-finetune-main/NASIL-CALISTIRILIR.md)

---

### 2. **MT5 Turkish Summarization — Türkçe Metin Özetleme** `mt5-turkish-summarization/`

Türkçe yazılı metinleri özet metinlere dönüştüren **mT5 (multilingual T5)** modelini fine-tuning ile optimize etme projesi.

**Temel Sonuç:**
- ✅ **Model:** mT5-small pretrained modeli Türkçe özet görevine adapte
- ✅ **Görev:** Uzun metinleri (örn. haber, makale, dokuman) kısa ve anlamlı özetlere dönüştürme
- ✅ **Veri:** Türkçe özet çiftleri ile fine-tune edilmiş
- ✅ **Donanım:** Ücretsiz Kaggle/Colab **T4 GPU** ile eğitime hazır

**Teknoloji:**
- Model: `ozcangundes/mt5-small-turkish-summarization` (Hugging Face hub'da)
- Mimari: **mT5** (Google'ın multilingual Transfer Transformer 5) — 101 dilde destek
- Yöntem: **Sequence-to-Sequence** (metin-metin dönüştürme)
- Veri: Türkçe metin-özet çiftleri
- Donanım: Ücretsiz T4 GPU

**Kullanım Alanları:**
- 📰 Haber özetleme (gazeteler, haber siteleri)
- 📄 Teknik dokümantasyon özeti
- 🔍 Arama motoru snippet'leri
- 📋 Toplantı notlarından özet çıkarma
- 📚 Akademik makalelerin özeti

**İçerik:**
| Dosya | Açıklama |
|---|---|
| `mt5_turkish_summarization.ipynb` | Ana notebook — modelin yüklenmesi, fine-tuning, inference |
| `google-colab-link` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/ozcangundes/mt5-small-turkish-summarization.ipynb) |

**Hızlı Örnek:**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-small-turkish-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-small-turkish-summarization")

metin = "Türkiye ekonomisi son yıllarda önemli değişimler geçirmektedir..."
inputs = tokenizer.encode(metin, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
ozet = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Özet: {ozet}")
```

**Hızlı Başlangıç (Colab):**
1. [Google Colab'da Aç](https://colab.research.google.com/#fileId=https%3A//huggingface.co/ozcangundes/mt5-small-turkish-summarization.ipynb)
2. **Runtime → Change runtime type → T4 GPU**
3. Hücreleri çalıştır → Model yüklenir + özetleme örneği yapılır
4. Kendi metinlerinizi test edin

👉 **Model Hub:** [ozcangundes/mt5-small-turkish-summarization](https://huggingface.co/ozcangundes/mt5-small-turkish-summarization)

---

### 3. **Car Prediction (Araba Fiyat Tahmini)** `car-predict/`

*Detaylı bilgi için dizin içindeki README'ye bakınız.*

---

### 4. **Fine-tuning Laboratuvarı** `Finetune/`

*Detaylı bilgi için dizin içindeki README'ye bakınız.*

---

## 🎯 Neden Gemma 3 4B?

Bu uygulama başta **Gemma 4 E4B** ile denenmiştir. Sonuç:

| Model | T4 VRAM | Durumu |
|---|---|---|
| Gemma 4 E4B (float16) | 8 GB → OOM | ❌ Yüklemede başarısız |
| Gemma 4 E2B (float16) | 6 GB → NaN | ❌ Eğitimde dtype hatası |
| **Gemma 3 4B** (float16) | 3.6 GB | ✅ %0 → %95 başarılı |

**Sebep:** T4 (Turing) bfloat16 desteklemez → Gemma 4 float16'da kararsız (NaN üretir) → float32 zorunlu → ücretsiz GPU'ya sığmaz.  
**Ders:** En yeni model her zaman en doğru seçim değildir. Doğru donanım seçimi önemlidir.

---

## 🛠 Teknik Stack

| Bileşen | Araç |
|---|---|
| **Dil** | Python 3.8+ |
| **Notebook** | Jupyter (`.ipynb`) — Kaggle/Colab çalıştırma |
| **Model** | Hugging Face `unsloth/gemma-3-4b-it`, `ozcangundes/mt5-small-turkish-summarization` |
| **Fine-tuning** | Unsloth (QLoRA), Hugging Face Transformers |
| **Kütüphaneler** | `torch`, `transformers`, `unsloth`, `peft`, `datasets` |
| **Veriler** | Hugging Face Datasets |
| **GPU** | Kaggle T4 (ücretsiz) veya Colab T4 (ücretsiz) |

---

## 📖 Dökümantasyon

- **`gemma3-json-finetune-main/README.md`** — Proje genel bakış + kavramlar  
- **`gemma3-json-finetune-main/NASIL-CALISTIRILIR.md`** — Adım adım çalıştırma (Kaggle + Colab) + hata çözümleri  
- **`gemma3-json-finetune-main/gemma3-json-finetune-sunu.pptx`** — Eğitim sunusu (PDF de mevcut)

---

## 🚀 Başlangıç

### Gereksinimleri İndir
```bash
pip install torch transformers unsloth peft datasets
```

### Notebook'u Çalıştır
**Kaggle (Önerilen — en stabil):**
1. https://www.kaggle.com → hesap oluştur + telefon doğrula
2. **Create → Notebook → Import**
3. `gemma3-json-finetune-main/gemma3_json_finetune.ipynb` seç
4. Sağ panel → **GPU T4 x2** + **Internet On**
5. **Run All**

**Google Colab:**
1. https://colab.research.google.com → **File → Upload notebook**
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**

---

## 📊 Sonuç Örneği

**Baseline (fine-tune öncesi):**
```
Giriş: "Fatura tarihi 15/03/2024, müşteri Ahmet, toplam 1500 TL"
Çıktı: "Lorem ipsum dolor sit amet..." (ham model gevezeliği)
Geçerli JSON: ❌ Hayır
```

**Fine-tuned (60 adım eğitim sonrası):**
```
Giriş: "Fatura tarihi 15/03/2024, müşteri Ahmet, toplam 1500 TL"
Çıktı: {"tarih": "15/03/2024", "müşteri": "Ahmet", "tutar": 1500}
Geçerli JSON: ✅ Evet
```

**Metrikler (20 test örneği):**
- Geçerli JSON oranı: %0 → %95
- Eğitim kaybı: 0.42 → 0.13
- İşlem süresi: ~15 dakika

---

## 🔗 Kaynaklar

- [Gemma Modelleri (Google)](https://huggingface.co/collections/google/gemma-release-65d746f89d895876a249ce11)
- [mT5 Model Dokümantasyonu](https://huggingface.co/docs/transformers/model_doc/mt5)
- [MT5 Turkish Summarization Model](https://huggingface.co/ozcangundes/mt5-small-turkish-summarization)
- [Unsloth Kütüphanesi](https://github.com/unslothai/unsloth)
- [LoRA Makalesi](https://arxiv.org/abs/2106.09685)
- [QLoRA Makalesi](https://arxiv.org/abs/2305.14314)
- [HuggingFace Datasets](https://huggingface.co/datasets)

---

## 📝 Lisans

Eğitim materyali (döküman + notebook). Model ve veri seti kendi lisanslarına tabiidir:
- **Gemma Modelleri:** Google tarafından (Community License)
- **mT5 Modeli:** Google tarafından (Apache 2.0)
- **Veri Seti:** `paraloq/json_data_extraction` (Apache 2.0)

---

## 🤝 Katkı & Soru

Sorular, hata bildirimleri veya iyileştirme önerileri için **Issues** açabilirsiniz.

---

**Eğitim Bilgisi**  
Antalya — Büyük Dil Modelleri (LLMs) Tabanlı Uygulama Geliştirme Eğitimi  
Dr. Murat Altun · 2024
