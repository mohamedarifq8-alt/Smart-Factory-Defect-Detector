# 📋 دليل تشغيل الـ Notebook بالمراحل السبع على Kaggle

تم تحديث دفتر الأكواد [`mvtec_yolo11_kaggle_training.ipynb`](file:///c:/Users/ALQURSAN/Desktop/تدريب%20نماذج%20ال%20AI/AI/mvtec_yolo11_kaggle_training.ipynb) ليحتوي على **7 خلايا مستقلة** يمكنك تشغيل كل خلية منها على حدة:

---

### جدول المراحل المستقلة داخل الـ Notebook:

| رقم الخلية | اسم المرحلة | ما الذي تقوم به؟ |
| :---: | :--- | :--- |
| **المرحلة 1** | **تثبيت المكتبات (Setup)** | تثبيت `ultralytics` و `onnx` والتحقق من كارت الشاشة GPU. |
| **المرحلة 2** | **اكتشاف مسار البيانات (Discovery)** | البحث التلقائي عن المنتجات الثلاثة (`bottle`, `metal_nut`, `capsule`) داخل `/kaggle/input`. |
| **المرحلة 3** | **تحويل البيانات (Conversion)** | قراءة أقنعة `ground_truth` وتحويلها آلياً لمربعات YOLO بدون أي تعديل يدوي. |
| **المرحلة 4** | **ملف التكوين (YAML Config)** | كتابة ملف إعدادات التدريب `mvtec_qc.yaml`. |
| **المرحلة 5** | **التدريب والحفظ الذكي (Training)** | تدريب `yolo11s` بنظام **Transfer Learning** مع: <br>• **حفظ تلقائي لأفضل أوزان `best.pt`** بعد كل دورة.<br>• **توقف مبكر ذكي (`patience=8`)** لإيقاف التدريب إذا لم يعد هناك تحسن. |
| **المرحلة 6** | **التصدير لأجهزة التحكم (ONNX Export)** | تحويل أفضل نموذج إلى صيغة `best_defect_detector.onnx` وضغط الحزمة في `industrial_qc_model_package.zip`. |
| **المرحلة 7** | **الاختبار المرئي (Visual Test)** | فحص 4 عينات حقيقية ورسم المربع الأحمر حول العيب وقرار (PASS / REJECT). |

---

### خطوات العمل في كاجل:
1. افتح موقع [Kaggle](https://www.kaggle.com) وارفع ملف `mvtec_yolo11_kaggle_training.ipynb`.
2. من القائمة اليمنى:
   - فعّل **GPU T4 x2**.
   - اضغط على **+ Add Input** وأضف: `https://www.kaggle.com/datasets/ipythonx/mvtec-ad`.
3. يمكنك تشغيل الخلية تلو الأخرى بالضغط على زر التشغيل (Play ▶️) بجانب كل مرحلة، أو الضغط على **Run All**.
4. بعد اكتمال المرحلة 6، ادخل إلى قسم **Output** في اليمين ونزل الملف:
   `industrial_qc_model_package.zip`.
