# KuşGöz – Derin Analiz (NovaFem)  https://kus.meydani.info

Video tabanlı yapay zekâ ile kuşların **otomatik tespiti**, **tür sınıflandırması**, **yaklaşık konum/irtifa** ve **popülasyon yoğunluğu** analizi. Hedef; havacılık güvenliğini artırmak, ekolojik araştırmaları hızlandırmak ve karar süreçlerine güvenilir veri sağlamaktır.

> **Proje ekibi (NovaFem):**  
> **Nisan Demiray** – Literatür, kavramsal tasarım, video çekimleri, veri etiketleme, model eğitimi, yazılım  
> **Meryemnur Pala** – Literatür, kavramsal tasarım, veri etiketleme, yazılım testi, görsel geliştirme  
> **Ezgi Kutlu** – Literatür, kavramsal tasarım, veri etiketleme  

---

## İçindekiler
- [Problem ve Değer Önerisi](#problem-ve-değer-önerisi)
- [Somut Hedefler](#somut-hedefler)
- [Yol Haritası](#yol-haritası)
- [Metodoloji](#metodoloji)
- [Teknik Yaklaşım ve Yeterlilik](#teknik-yaklaşım-ve-yeterlilik)
- [Proje Çıktıları](#proje-çıktıları)
- [Etkiler](#etkiler)
- [Gerçek Dünya Uygulamaları](#gerçek-dünya-uygulamaları)
- [Kurulum & Çalıştırma (örnek)](#kurulum--çalıştırma-örnek)
- [İletişim](#iletişim)
- [Lisans](#lisans)

---

## Problem ve Değer Önerisi
- **Mevcut yöntemler** (radar/insan gözlemi) pahalı, zaman alıcı ve sınırlı kapsama sahip.  
- **KuşGöz**, sabit kamera ve drone verisinden **otomatik, gerçek zamanlı ve veriye dayalı** analiz üretir.  
- **Çift yönlü değer:**  
  - **Ekosistem:** Göç yolları, popülasyon dinamikleri ve etolojik içgörü.  
  - **Güvenlik:** Havacılık/savunma için proaktif risk yönetimi.

---

## Somut Hedefler
- Kuşları **>%90 doğruluk** ile tespit etmek.  
- Tespit edilen kuşları **tür bazında ≥%85 doğruluk** ile sınıflandırmak.  
- Konum/irtifa & sürü yoğunluğu için **çok katmanlı analiz** üretmek.  

---

## Yol Haritası
- **Kısa vade:** Veri toplama → etiketleme → model eğitim/validasyon → arayüz (frame-frame oynatım, hızlı atlama, stream/video desteği).  
- **Orta vade:** Çoklu kamera/drone entegrasyonu, saha testleri.  
- **Gelecek vizyonu:**  
  - **Hava taksileri (eVTOL) gerçeği**  
  - **UTM (hava trafik yönetimi) ihtiyacı**  
  - **Büyüyen ekolojik veri talebi**  
  - **İHA savunma sistemleri ile entegrasyon**  
  - **RES (rüzgâr) projelerinde raporlama süreciyle uyum**

---

## Metodoloji
1. **Veri seti oluşturma:** Sabit kamera & drone çekimleri.  
2. **Etiketleme:** **CVAT** üzerinde sınıf/box anotasyonları.  
3. **Ortam:** **Colab + Google Drive**.  
4. **Ön işleme:** Veri konsolidasyonu, temizlik, **%80/%20** eğitim/validasyon ayrımı.  
5. **Model eğitimi:** **YOLOv11n**, **1920** çözünürlük, **100 epoch**.  
6. **Test & Devreye Alım:** Modelin yazılıma gömülmesi ve arayüzle entegrasyon.

---

## Teknik Yaklaşım ve Yeterlilik
- **Küçük nesne filtresi:** Eğitim öncesi **80 piksel² altı** nesneleri eleme.  
- **Mesafe/irtifa tahmini:** **Geometrik ortalama** tabanlı model.  
- **Takip & düzgünleştirme:** **Son 5 görünüm** üzerinden mesafe düzeltmesi.  
- **Kalıcılık metriği:** Nesnenin ardışık kaç frame’de göründüğünün hesabı.  
- **Sürü tespiti:** Aynı tip ≥3 nesneden **küçük/kararsız** objelere genişletme.  
- **Parametrik mimari:** Tüm algoritmalar **ayarlanabilir**.  
- **Arayüz:** Frame-frame oynatma, doğrudan hedef frame’e atlama, hızlı atlama; **kayıt ve stream** kaynaklarıyla çalışma.

---

## Proje Çıktıları
- **Tespit & sınıflandırma** sonuçları (video üstü çizim/overlay).  
- **Sürüler ve yoğunluk haritaları** (zamana/konuma bağlı).  
- **Risk göstergeleri** (hava sahası güvenliği için).  
- **Raporlama/özet** çıktıları (operasyon ekiplerine uygun).

---

## Etkiler
- **Sosyal & Güvenlik:** Hava sahası güvenliği, güvenli seyahat, can/mal emniyeti.  
- **Ekonomik:** Maliyet azaltma, operasyonel verimlilik.  
- **Çevresel & Bilimsel:** Biyoçeşitliliğin korunması, sürdürülebilirlik, araştırma verimliliği.

---

## Gerçek Dünya Uygulamaları
- **Havaalanları:** Kuş aktivitesinin gerçek zamanlı izlenmesi; riskli durumların erken tespiti.  
- **Ekolojik araştırmalar:** Göçmen tür takibi, insan-doğa etkileşimi.  
- **Enerji sektörü (RES):** Çarpışma riskinin azaltılması, izleme & uyum raporları.  
- **Savunma:** Artan **SİHA tehdidine** karşı tespit güvenilirliğinin artırılması.

---

## Kurulum & Çalıştırma (örnek)
> Bu bölüm, projeyi koda taşıdığınızda güncellenebilir. Aşağıdaki iskelet, tipik bir Python/YOLO tabanlı kurulum içindir.

```bash
# 1) Ortam
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Model ağırlıklarını yerleştir
# weights/ klasörüne yolov8n*.pt koyun (ya da training pipeline ile üretin)

# 3) Çalıştırma (örnek)
python app.py --source path/to/video_or_stream --conf 0.25 --imgsz 640
```

**Dizin önerisi**
```
.
├─ data/               # örnek videolar / anotasyonlar
├─ weights/            # YOLO ağırlıkları
├─ src/                # model çağrıları, izleme, metrikler
├─ ui/                 # arayüz (stream/video oynatıcı, overlay)
└─ app.py              # giriş noktası
```

---

## İletişim
- **Nisan Demiray** – n.demiray2018@gtu.edu.tr – linkedin.com/in/nisan-kandemir  
- **Meryemnur Pala** – meryemnur6969@gmail.com – linkedin.com/in/meryemnur-pala  
- **Ezgi Kutlu** – ezgikutlu72@gmail.com – linkedin.com/in/ezgi-kutlu

---

## Lisans
Bu depo altındaki kod ve dokümanların lisansı, `LICENSE` dosyasındaki hükümlere tabidir (eklendiğinde).
