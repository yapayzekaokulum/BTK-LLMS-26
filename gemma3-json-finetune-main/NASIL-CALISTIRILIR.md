# Gemma 3 (4B) Fine-Tuning Notebook'u — Nasıl Çalıştırılır?

**Antalya LLM Tabanlı Uygulama Geliştirme Eğitimi · Dr. Murat Altun**

Bu rehber, `gemma3_json_finetune.ipynb` dosyasını **ücretsiz** bir GPU üzerinde uçtan uca çalıştırmayı adım adım anlatır. Kod bilmenize gerek yok — hücreleri sırayla çalıştırmanız yeterli.

> **Notebook ne yapıyor?** Dağınık bir belgeyi (fatura, tıbbi kayıt, sipariş…) + bir JSON şemasını alıp, şemaya birebir uyan **geçerli JSON** üreten bir model eğitir. **Gemma 3 4B**'yi **QLoRA** ile tek ücretsiz T4 GPU'da fine-tune ederiz.

> ✅ **Kaggle T4'te gerçek sonuç:** Geçerli-JSON oranı fine-tune öncesi **%0** → sonrası **%95**. Yükleme 4.6 GB, eğitim peak VRAM 5.9 GB.

---

## 0. Önce kısa kavramlar (LoRA / QLoRA nedir?)

| Kavram | Bir cümlede |
|---|---|
| **Base vs Instruct (-it)** | Base model ham cümle tamamlayıcıdır; **Instruct** (bizim kullandığımız `-it`) komut anlar, cevap verir. |
| **Quantization (4-bit)** | Modelin 16-bit hali ~8 GB; **4-bit**'e sıkıştırınca ~3 GB → ücretsiz GPU'ya sığar, doğrulukta ihmal edilebilir kayıp. |
| **LoRA** | Donuk modeli baştan yazmak yerine kenara küçük **"not kağıtları"** iliştiririz. Model aynı kalır, notlar görevi öğrenir. |
| **QLoRA** | **LoRA + 4-bit taban.** Sadece parametrenin ~%0.5'ini eğitiriz → ucuz, hızlı, ücretsiz GPU'da çalışır. |
| **SFT** | *Supervised Fine-Tuning* — modele "bu girdiye şu çıktı" örneklerini göstererek öğretmek. |

---

## A. Kaggle'da Çalıştırma (önerilen — en stabil ücretsiz GPU)

### A.1. Hesap ve GPU hakkı
1. [kaggle.com](https://www.kaggle.com) → ücretsiz hesap (Google ile girilebilir).
2. **Telefon doğrulaması** yap (Settings → Phone Verification) — bu olmadan GPU açılmaz. Sonrası haftalık **~30 saat ücretsiz GPU**.

### A.2. Notebook'u yükle
1. **Create → New Notebook** → **File → Import Notebook**.
2. **`gemma3_json_finetune.ipynb`**'yi seç / sürükle.

### A.3. GPU ve İnterneti aç (KRİTİK)
Sağ panel → **Session options**:
- **Accelerator** → **GPU T4 x2**
- **Internet** → **On** *(model + veri indirilecek; kapalıysa çalışmaz!)*

### A.4. Çalıştır
- Üstten **Run All**.
- **İlk 2–4 dk:** Unsloth kurulumu + model (~3 GB) indirme.
- **Sonra:** baseline ölçümü → 60 adım eğitim (~10 dk) → sonrası ölçümü.

### A.5. Sonucu gör
- **Baseline** hücresi: fine-tune *öncesi* geçerli-JSON oranı (düşük — model gevezelik yapar).
- **Sonrası** hücresi: fine-tune *sonrası* oran (yüksek — tertemiz JSON).
- Referans çalıştırmada: **%0 → %95**.

### A.6. Modeli indir (opsiyonel)
- `gemma3-json-lora/` klasörü oluşur → sağdaki **Output** sekmesinden indir / Kaggle'da yayınla.

---

## B. Google Colab'da Çalıştırma (alternatif)

1. [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.**

> Notebook Colab ile Kaggle'ı otomatik ayırt eder — sadece Run All yeterli.

---

## C. Sık Karşılaşılan Hatalar ve Çözümleri

| Belirti | Sebep | Çözüm |
|---|---|---|
| `No CUDA GPUs are available` | GPU seçilmemiş | Accelerator = **GPU T4**, oturumu yeniden başlat |
| İndirme takılıyor / `ConnectionError` | İnternet kapalı | **Internet = On** |
| `gated repo` / `401` | Model erişimi | Kaggle → Add-ons → Secrets → `HF_TOKEN` ekle; notebook notundaki `login(...)`'ı çalıştır |
| `CUDA out of memory` | Bellek doldu | `max_seq_length`'i 2048 → 1024 düşür **veya** oturumu Restart edip baştan çalıştır |
| `NameError` / `... not defined` | Hücreler sırasız çalıştı | Baştan **Run All** (yukarıdan aşağı) |
| Ölçüm çok uzun sürüyor | Baseline'da ham model her örnekte 512 token gevezelik yapıyor | Normal; isterseniz baseline örnek sayısını (20) veya `max_new_tokens`'ı düşürün |

---

## D. Çıktıyı Nasıl Yorumlarız?

1. **Loss** eğitim boyunca **düşüyorsa** model JSON kalıbını öğreniyordur (referansta 0.42 → 0.13).
2. **Baseline oranı** (referansta %0) düşük, **fine-tuned oranı** (referansta %95) yüksek olmalı — bu *iyileşmenin objektif kanıtı*.
3. Örnek çıktıda `json.loads()` **hatasız** çalışıyorsa → çıktı gerçekten geçerli JSON'dur.

> **Bilimsel disiplin:** "Sanırım iyi oldu" demiyoruz. Öncesini ve sonrasını ayrı test setinde ölçüp kıyaslıyoruz (data leakage yok).

---

## E. Model seçim notu — niçin Gemma 3 4B, Gemma 4 E4B değil?

Bu uygulama önce Gemma 4 E4B ile denendi; ücretsiz Kaggle T4'te **eğitilemedi**:
- **T4 (Turing) bfloat16 desteklemez.** Gemma 4 float16'da kararsızdır (NaN) → float32 zorunlu → 8B model 14.5 GB'a sığmaz (OOM).
- **Gemma 3 4B** float16'da stabildir → ücretsiz T4'te sorunsuz eğitilir (kanıt: %0 → %95).

Ders: *En yeni model her zaman en doğru seçim değildir.* Gemma 4 QLoRA için bf16'lı Ampere+ GPU (L4/A100, ücretli) gerekir; ücretsiz eğitim için doğru seçim **Gemma 3 4B**.

---

## F. Sonraki Adımlar (öğrenci için)

- `max_steps=60` yerine `num_train_epochs=2` ile **tam eğitim**; oranı tekrar ölç.
- `jsonschema.validate` ile şemaya uygunluk + alan F1.
- **Kendi Türkçe verinle** (fatura/dilekçe → JSON) aynı hattı çalıştır — gerçek dünyada asıl beceri budur.
- **GGUF**'a çevirip **Ollama** ile yerelde çalıştır (offline, ücretsiz).

---

*Kaynak veri seti: `paraloq/json_data_extraction` (Hugging Face, apache-2.0) — notebook çalışırken otomatik iner.*
