# Forex Trend Prediction System

ระบบทำนายแนวโน้มของค่าเงินในตลาด Forex โดยใช้ Machine Learning และ Deep Learning

## ภาพรวมของระบบ

ระบบนี้ได้รับการออกแบบเพื่อทำนายราคาในตลาด Forex โดยใช้เทคนิคต่างๆ ดังนี้:

1. ใช้โมเดลหลายประเภท: LSTM, GRU, XGBoost, และ TFT (Temporal Fusion Transformer)
2. ปรับปรุงประสิทธิภาพโดยการเพิ่ม Technical Indicators และการคัดเลือกคุณลักษณะ
3. ใช้เทคนิค Bagging Approach ที่รวมข้อมูลจากหลายคู่เงินเพื่อปรับปรุงความแม่นยำ

ระบบแบ่งออกเป็น 4 ส่วนหลัก:

1. **Data Acquisition Stage**: โหลดและเตรียมข้อมูล
2. **Feature Engineering Stage**: เพิ่ม Technical Indicators และทำ Feature Selection
3. **Prediction Stage**: เทรนโมเดลและทำนายราคา
4. **Evaluation Stage**: ประเมินผลโมเดล

## โครงสร้างโปรเจค

```
forex_prediction/
│
├── config/
│   └── config.py          # ตั้งค่าพารามิเตอร์ต่างๆ
│
├── data/
│   ├── EURUSD_1H.csv      # ข้อมูลคู่เงิน EURUSD
│   ├── GBPUSD_1H.csv      # ข้อมูลคู่เงิน GBPUSD
│   └── USDJPY_1H.csv      # ข้อมูลคู่เงิน USDJPY
│
├── models/                # โฟลเดอร์เก็บโมเดลที่เทรนแล้ว
│
├── results/               # โฟลเดอร์เก็บผลลัพธ์
│
├── logs/                  # โฟลเดอร์เก็บ log ต่างๆ
│
├── src/
│   ├── __init__.py
│   ├── stage1_data_acquisition.py
│   ├── stage2_feature_engineering.py
│   ├── stage3_prediction_models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   ├── xgboost_model.py
│   │   └── tft_model.py
│   ├── stage4_evaluation.py
│   ├── visualization.py
│   └── reporting.py
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── evaluation_utils.py
│
└── main.py                # เริ่มต้นรันโปรแกรม
```

## การติดตั้ง

### ความต้องการของระบบ

- Python 3.8 หรือสูงกว่า
- TensorFlow 2.x
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### วิธีการติดตั้ง

1. Clone repository:
   ```bash
   git clone https://github.com/your-username/forex-prediction.git
   cd forex-prediction
   ```

2. สร้าง virtual environment และติดตั้ง dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # สำหรับ Linux/Mac
   # หรือ
   venv\Scripts\activate     # สำหรับ Windows
   
   pip install -r requirements.txt
   ```

3. เตรียมข้อมูล:
   - วางไฟล์ `EURUSD_1H.csv`, `GBPUSD_1H.csv`, และ `USDJPY_1H.csv` ในโฟลเดอร์ `data/`

## การใช้งาน

รันโปรแกรมในโหมดต่างๆ:

```bash
# รันทั้งระบบ
python main.py --mode=all

# รันเฉพาะส่วนโหลดข้อมูล
python main.py --mode=data

# รันเฉพาะส่วน Feature Engineering
python main.py --mode=features

# รันเฉพาะการเทรนโมเดล
python main.py --mode=train

# รันเฉพาะการประเมินผล
python main.py --mode=evaluate

# สร้างกราฟและภาพ
python main.py --mode=visualize

# สร้างรายงานสรุป
python main.py --mode=report
```

ตัวเลือกเพิ่มเติม:

```bash
# เลือกคู่เงินเฉพาะ
python main.py --pair=EURUSD

# เลือกโมเดลเฉพาะ
python main.py --model=LSTM

# รวมตัวเลือกหลายอย่าง
python main.py --mode=train --pair=EURUSD --model=LSTM
```

## แนวคิดของระบบ

ระบบนี้ใช้แนวทางใหม่ในการทำนายราคา Forex โดยเน้นที่:

1. **การเปรียบเทียบประสิทธิภาพระหว่างชุดข้อมูลต่างๆ**:
   - Raw Data [..1]: ข้อมูลดิบไม่มีการปรับแต่ง
   - Enhanced Data [..2]: เพิ่ม Technical Indicators
   - Enhanced + Selected [..3]: เพิ่ม Technical Indicators และทำ Feature Selection

2. **การเปรียบเทียบโมเดลต่างๆ**:
   - LSTM: Long Short-Term Memory networks
   - GRU: Gated Recurrent Units
   - XGBoost: Extreme Gradient Boosting
   - TFT: Temporal Fusion Transformer

3. **Bagging Approach**:
   - ใช้ข้อมูลจากทั้ง 3 คู่เงิน (EURUSD, GBPUSD, USDJPY) มาเทรนโมเดลร่วมกัน
   - ช่วยให้โมเดลเรียนรู้แพทเทิร์นที่เป็นสากลในตลาด Forex
   - ลดการ Overfit ต่อแพทเทิร์นเฉพาะของคู่เงินใดคู่เงินหนึ่ง

4. **การประเมินผล**:
   - Statistical Metrics: RMSE, MAE, MAPE
   - Financial Performance: Annualized Return, Sharpe Ratio, Max Drawdown
   - Trading Effectiveness: Win Rate, Risk-Reward Ratio
   - Benchmark Comparison: เทียบกับกลยุทธ์ Buy-and-Hold, SMA Crossover, และ Random

## การปรับแต่งพารามิเตอร์

แก้ไขค่าพารามิเตอร์ต่างๆ ได้ที่ไฟล์ `config/config.py`:

- ช่วงเวลาสำหรับเทรนและทดสอบ (`TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END`)
- โมเดลพารามิเตอร์ (`LSTM_PARAMS`, `GRU_PARAMS`, `XGB_PARAMS`, `TFT_PARAMS`)
- Technical Indicators ที่ใช้ (`TECHNICAL_INDICATORS`)
- พารามิเตอร์สำหรับการเทรด (`INITIAL_BALANCE`, `TRADING_COMMISSION`)

## ผลลัพธ์และรายงาน

หลังจากรันระบบ ผลลัพธ์จะถูกบันทึกในโฟลเดอร์ `results/`:

- กราฟเปรียบเทียบราคาจริงและราคาที่ทำนาย
- กราฟเปรียบเทียบเมตริกต่างๆ ระหว่างโมเดล
- กราฟเปรียบเทียบผลลัพธ์ระหว่างประเภทข้อมูล
- กราฟเปรียบเทียบผลลัพธ์ระหว่าง Single Pair และ Bagging Approach
- รายงานสรุปในรูปแบบ Markdown

## สนับสนุนหรือติดต่อ

หากมีคำถามหรือต้องการสนับสนุนโปรเจคนี้ กรุณาติดต่อผ่าน:
- Email: your.email@example.com
- GitHub Issues: [https://github.com/your-username/forex-prediction/issues](https://github.com/your-username/forex-prediction/issues)