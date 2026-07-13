# Gemma 3 (4B) Fine-Tuning — Metinden Yapılandırılmış JSON Çıkarma

Dağınık bir belgeyi (fatura, tıbbi kayıt, sipariş…) + bir JSON şemasını alıp, şemaya birebir uyan **geçerli JSON** üreten bir modeli **QLoRA** ile **ücretsiz Kaggle T4 GPU** üzerinde fine-tune eden uçtan uca eğitim materyali.

**Antalya — Büyük Dil Modelleri (LLMs) Tabanlı Uygulama Geliştirme Eğitimi · Dr. Murat Altun**

---

## 🎯 Sonuç (Kaggle T4'te gerçek çalıştırma)

| Metrik | Değer |
|---|---|
| Geçerli-JSON oranı (fine-tune **öncesi**) | **%0** |
| Geçerli-JSON oranı (fine-tune **sonrası**) | **%95** |
| Eğitim kaybı (loss) | 0.42 → 0.13 |
| Yükleme VRAM | 4.6 GB |
| Eğitim peak VRAM | 5.9 GB (T4'ün 14.5 GB'ına rahat sığar) |

> 20 örneklik ayrı test setinde ölçüldü. Ham model 512 token gevezelik üretip hiç geçerli JSON veremezken, 60 adımlık minik eğitim sonrası 20 örneğin 19'u tertemiz JSON.

---

## 📦 İçerik

| Dosya | Açıklama |
|---|---|
| [`gemma3_json_finetune.ipynb`](gemma3_json_finetune.ipynb) | Ana notebook — kavramlar + kod + baseline/sonrası ölçüm (Kaggle/Colab) |
| [`gemma3-json-finetune-sunu.pptx`](gemma3-json-finetune-sunu.pptx) | 11 slaytlık eğitim sunusu (konuşmacı notlarıyla) + PDF |
| [`NASIL-CALISTIRILIR.md`](NASIL-CALISTIRILIR.md) | Adım adım çalıştırma rehberi (Kaggle + Colab + sık hatalar) + PDF |

## 🧱 Yığın

- **Model:** [`unsloth/gemma-3-4b-it`](https://huggingface.co/unsloth/gemma-3-4b-it) (Google açık ağırlık)
- **Yöntem:** QLoRA (4-bit) · **Kütüphane:** [Unsloth](https://github.com/unslothai/unsloth)
- **Veri:** [`paraloq/json_data_extraction`](https://huggingface.co/datasets/paraloq/json_data_extraction) (484 örnek, apache-2.0)
- **Donanım:** Ücretsiz Kaggle / Colab **T4** GPU

## 🚀 Hızlı Başlangıç

1. [kaggle.com](https://www.kaggle.com) → hesap + telefon doğrulama (GPU için).
2. **Create → Notebook → Import** → `gemma3_json_finetune.ipynb`.
3. Sağ panel: **Accelerator = GPU T4 x2**, **Internet = On**.
4. **Run All** → ~15 dk sonra baseline %0 → sonrası %95 sonucu ekrana düşer.

Ayrıntı: [NASIL-CALISTIRILIR.md](NASIL-CALISTIRILIR.md).

## 🔍 Niçin Gemma 3 4B, Gemma 4 E4B değil?

Bu uygulama önce Gemma 4 E4B ile denendi; **ücretsiz T4'te eğitilemedi** (13 gerçek çalıştırma):

| Model | Sonuç |
|---|---|
| Gemma 4 E4B (auto/float16/QAT) | ❌ Yüklemede OOM |
| Gemma 4 E2B | ❌ Eğitimde dtype hatası / OOM |
| **Gemma 3 4B** | ✅ Eğitildi: %0 → %95 |

**Kök neden:** T4 (Turing) bfloat16 desteklemez → Gemma 4 float16'da NaN üretir → float32 zorunlu → 8B model belleğe sığmaz. Gemma 3 4B float16'da stabildir. *En yeni model her zaman en doğru seçim değildir; donanımınla modelin gereksinimi uyuşmalı.* Gemma 4 QLoRA için bf16'lı Ampere+ (L4/A100, ücretli) gerekir.

## 📄 Lisans

Eğitim materyali. Model & veri seti kendi lisanslarına tabidir (Gemma / apache-2.0).
